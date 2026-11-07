/* Raga Search: browser client for the /identify API.
 *
 * Records raw PCM from the microphone (with echo cancellation, noise
 * suppression and auto gain off, since they damage music), and every few seconds
 * sends everything recorded so far as a 16-bit WAV. Each result replaces
 * the previous one, so the guess sharpens as more music is heard.
 */
(() => {
  "use strict";

  const FIRST_GUESS_S = 8;   // first request after this much audio
  const INTERVAL_S = 5;      // then every INTERVAL_S seconds
  const MAX_S = 60;          // stop listening automatically
  const TARGET_SR = 22050;   // upload sample rate (plenty for pitch tracking)
  const SVARAS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"];
  const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

  const $ = (id) => document.getElementById(id);
  const els = {
    listenBtn: $("listenBtn"), listenLabel: $("listenLabel"), progress: $("progressBar"),
    status: $("status"), fileInput: $("fileInput"), error: $("error"),
    results: $("results"), bestName: $("bestName"), bestConf: $("bestConf"), bestThaat: $("bestThaat"),
    bestTime: $("bestTime"), bestSvaras: $("bestSvaras"), others: $("others"), confHint: $("confHint"),
    saCard: $("saCard"), saNote: $("saNote"), saHz: $("saHz"), saSource: $("saSource"),
    saDown: $("saDown"), saUp: $("saUp"), saPlay: $("saPlay"), saReset: $("saReset"),
    svaraCard: $("svaraCard"), svaraChart: $("svaraChart"), ragaList: $("ragaList"), ragaCount: $("ragaCount"),
  };

  const state = {
    rec: null,            // active recording
    lastAudio: null,      // Blob most recently analysed (for re-analysis with a corrected Sa)
    lastName: "clip.wav",
    tonicOverride: null,  // Hz, set by the user
    lastResult: null,
    inFlight: false,
    requestId: 0,
    drone: null,
  };

  // ------------------------------------------------------------------ helpers

  const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);
  const hzToMidi = (hz) => Math.round(69 + 12 * Math.log2(hz / 440));
  const SA_MIN_MIDI = 43, SA_MAX_MIDI = 67; // G2 (98 Hz) .. G4 (392 Hz)

  function buildSaPicker() {
    for (let m = SA_MIN_MIDI; m <= SA_MAX_MIDI; m++) {
      const opt = document.createElement("option");
      opt.value = String(m);
      opt.textContent = noteName(midiToHz(m)).name;
      els.saNote.append(opt);
    }
  }

  function showSa(hz, source) {
    const m = Math.min(SA_MAX_MIDI, Math.max(SA_MIN_MIDI, hzToMidi(hz)));
    els.saNote.value = String(m);
    const { cents } = noteName(hz);
    els.saHz.textContent = `${hz.toFixed(1)} Hz${Math.abs(cents) >= 10 ? ` (${cents > 0 ? "+" : ""}${cents}¢)` : ""}`;
    els.saSource.textContent = source === "user" ? "set by you" : "detected";
    els.saReset.hidden = source !== "user";
  }

  function fmtTime(s) {
    s = Math.max(0, Math.floor(s));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function noteName(hz) {
    const midi = 69 + 12 * Math.log2(hz / 440);
    const n = Math.round(midi);
    const cents = Math.round((midi - n) * 100);
    const name = NOTE_NAMES[((n % 12) + 12) % 12] + (Math.floor(n / 12) - 1);
    return { name, cents };
  }

  function showError(msg) {
    els.error.textContent = msg;
    els.error.hidden = !msg;
  }

  function setStatus(msg) { els.status.textContent = msg; }

  function currentTonic() {
    return state.tonicOverride ?? state.lastResult?.tonic_hz ?? null;
  }

  // ------------------------------------------------------------------ audio encoding

  function downsample(samples, fromSr, toSr) {
    if (fromSr === toSr) return samples;
    const ratio = fromSr / toSr;
    const out = new Float32Array(Math.floor(samples.length / ratio));
    for (let i = 0; i < out.length; i++) {
      // Box filter over the source window: cheap anti-aliasing.
      const start = Math.floor(i * ratio), end = Math.min(samples.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j++) sum += samples[j];
      out[i] = sum / Math.max(1, end - start);
    }
    return out;
  }

  function encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function recordingAsWav(rec) {
    const total = rec.chunks.reduce((n, c) => n + c.length, 0);
    const all = new Float32Array(total);
    let off = 0;
    for (const c of rec.chunks) { all.set(c, off); off += c.length; }
    return encodeWav(downsample(all, rec.ctx.sampleRate, TARGET_SR), TARGET_SR);
  }

  // ------------------------------------------------------------------ recording

  const WORKLET = `
    class Capture extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) this.port.postMessage(ch.slice(0));
        return true;
      }
    }
    registerProcessor("capture", Capture);`;

  async function startRecording() {
    showError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("This browser can't use the microphone here. Open the app over HTTPS (or on localhost), or analyse a recording instead.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
    } catch (e) {
      showError(e.name === "NotAllowedError"
        ? "Microphone permission was denied. Allow it in your browser settings, or analyse a recording instead."
        : `Couldn't start the microphone (${e.message}).`);
      return;
    }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(stream);
    const rec = { ctx, stream, source, chunks: [], samples: 0, level: 0, node: null, timer: null, nextSendAt: FIRST_GUESS_S };

    const onChunk = (data) => {
      rec.chunks.push(data);
      rec.samples += data.length;
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      rec.level = 0.8 * rec.level + 0.2 * Math.min(1, Math.sqrt(sum / data.length) * 6);
    };

    if (ctx.audioWorklet) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      rec.node = new AudioWorkletNode(ctx, "capture");
      rec.node.port.onmessage = (e) => onChunk(e.data);
    } else {
      rec.node = ctx.createScriptProcessor(4096, 1, 1);
      rec.node.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
    }
    // Route through a muted gain so the graph keeps pulling audio without playing it back.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(rec.node).connect(mute).connect(ctx.destination);

    state.rec = rec;
    state.lastResult = null;
    state.tonicOverride = null;
    clearResults();
    els.listenBtn.setAttribute("aria-pressed", "true");
    els.listenLabel.textContent = "Stop listening";
    rec.timer = setInterval(tick, 200);
    tick();
  }

  function elapsed(rec) { return rec.samples / rec.ctx.sampleRate; }

  function tick() {
    const rec = state.rec;
    if (!rec) return;
    const t = elapsed(rec);
    els.progress.style.width = `${Math.min(100, (t / MAX_S) * 100)}%`;
    els.listenBtn.style.setProperty("--level", rec.level.toFixed(3));
    if (t >= MAX_S) { stopRecording(); return; }
    if (t >= rec.nextSendAt && !state.inFlight) {
      rec.nextSendAt = t + INTERVAL_S;
      analyse(recordingAsWav(rec), "live.wav", { live: true });
    }
    if (!state.inFlight) {
      setStatus(state.lastResult
        ? `Listening ${fmtTime(t)} · updating every ${INTERVAL_S} s`
        : `Listening ${fmtTime(t)} · first guess at ${fmtTime(FIRST_GUESS_S)}`);
    }
  }

  async function stopRecording() {
    const rec = state.rec;
    if (!rec) return;
    state.rec = null;
    clearInterval(rec.timer);
    rec.stream.getTracks().forEach((tr) => tr.stop());
    rec.source.disconnect();
    rec.node.disconnect();
    const t = elapsed(rec);
    const wav = t >= 3 ? recordingAsWav(rec) : null;
    await rec.ctx.close();
    els.listenBtn.setAttribute("aria-pressed", "false");
    els.listenLabel.textContent = "Start listening";
    els.listenBtn.style.setProperty("--level", 0);
    if (wav) {
      await analyse(wav, "recording.wav", { final: true });
    } else {
      setStatus("Too short. Listen for at least a few seconds.");
    }
  }

  // ------------------------------------------------------------------ API

  async function analyse(blob, name, { live = false, final = false } = {}) {
    state.lastAudio = blob;
    state.lastName = name;
    const id = ++state.requestId;
    state.inFlight = true;
    if (!live) { setStatus("Analysing…"); els.listenBtn.classList.add("busy"); }
    const form = new FormData();
    form.append("audio", blob, name);
    form.append("k", "3");
    if (state.tonicOverride) form.append("tonic_hz", state.tonicOverride.toFixed(2));
    try {
      const res = await fetch("identify", { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (id !== state.requestId) return; // a newer request superseded this one
      if (!res.ok) {
        const detail = typeof body.detail === "string" ? body.detail : `server error ${res.status}`;
        if (live) { setStatus(`Listening… (${detail})`); } else { showError(detail); setStatus(""); }
        return;
      }
      showError("");
      state.lastResult = body;
      render(body);
      if (!state.rec) {
        setStatus(final || !live
          ? `Analysed ${fmtTime(body.duration_s)} of audio.`
          : "");
      }
    } catch (e) {
      if (id === state.requestId) showError("Couldn't reach the server. Check your connection and try again.");
    } finally {
      if (id === state.requestId) { state.inFlight = false; els.listenBtn.classList.remove("busy"); }
    }
  }

  // ------------------------------------------------------------------ rendering

  function clearResults() {
    els.results.hidden = els.saCard.hidden = els.svaraCard.hidden = true;
    stopDrone();
  }

  function render(r) {
    const [best, ...others] = r.predictions;
    els.results.hidden = els.saCard.hidden = els.svaraCard.hidden = false;

    els.bestName.textContent = best.display;
    els.bestConf.textContent = `${Math.round(best.confidence * 100)}%`;
    els.bestThaat.textContent = best.thaat || "–";
    els.bestTime.textContent = best.time || "–";
    els.bestSvaras.textContent = best.svaras || "–";

    els.others.replaceChildren(...others.map((p) => {
      const li = document.createElement("li");
      const name = document.createElement("span"); name.className = "name"; name.textContent = p.display;
      const pct = document.createElement("span"); pct.className = "pct"; pct.textContent = `${Math.round(p.confidence * 100)}%`;
      const bar = document.createElement("div"); bar.className = "bar";
      const fill = document.createElement("div"); fill.style.width = `${Math.max(2, p.confidence * 100)}%`;
      bar.append(fill);
      li.append(name, pct, bar);
      return li;
    }));

    const hints = [];
    if (state.rec && r.duration_s < 30) hints.push("Early guess. It usually improves up to about 60 s.");
    if (best.confidence < 0.4) hints.push("Low confidence. The raga may not be one this prototype knows, or the Sa may be off.");
    // Performers keep returning to Sa, so very little time on S suggests a wrong tonic.
    if ((r.svara_profile.S || 0) < 0.06 && r.tonic_source !== "user") {
      hints.push("Very little time was spent on Sa, so the detected Sa is probably wrong. Play the drone below and try another note.");
    }
    if (r.voiced_fraction < 0.4) hints.push("Not much clear melody heard. Move closer to the music if you can.");
    els.confHint.textContent = hints.join(" ");
    els.confHint.hidden = hints.length === 0;

    // Sa
    showSa(r.tonic_hz, r.tonic_source);
    updateDrone();

    // Svara profile
    const inScale = new Set((best.svaras || "").split(/\s+/).filter(Boolean));
    const values = SVARAS.map((s) => r.svara_profile[s] || 0);
    const max = Math.max(...values, 1e-9);
    els.svaraChart.replaceChildren(...SVARAS.map((s, i) => {
      const col = document.createElement("div");
      col.className = "svara-col" + (inScale.has(s) ? " in" : "");
      const b = document.createElement("div"); b.className = "b";
      b.style.height = `${(values[i] / max) * 100}%`;
      b.title = `${s}: ${(values[i] * 100).toFixed(1)}%`;
      const label = document.createElement("span"); label.textContent = s;
      col.append(b, label);
      return col;
    }));
    els.svaraChart.setAttribute("aria-label",
      "Time spent on each svara: " + SVARAS.map((s, i) => `${s} ${(values[i] * 100).toFixed(0)}%`).join(", "));
  }

  // ------------------------------------------------------------------ Sa correction

  function shiftSa(semitones) {
    const t = currentTonic();
    if (!t) return;
    state.tonicOverride = t * Math.pow(2, semitones / 12);
    reanalyseWithSa();
  }

  function reanalyseWithSa() {
    if (state.rec) {
      state.rec.nextSendAt = 0; // next tick sends with the new Sa
      setStatus("Updating with the new Sa…");
    } else if (state.lastAudio) {
      analyse(state.lastAudio, state.lastName);
    }
    // Optimistic UI so the controls feel responsive.
    if (state.tonicOverride) {
      showSa(state.tonicOverride, "user");
      updateDrone();
    }
  }

  // A simple tanpura-like drone (low Pa, Sa, Sa, low Sa) to check Sa by ear.
  function startDrone() {
    const t = currentTonic();
    if (!t) return;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const out = ctx.createGain();
    out.gain.setValueAtTime(0, ctx.currentTime);
    out.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.3);
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 1800;
    lp.connect(out).connect(ctx.destination);
    const ratios = [0.75, 1, 1, 0.5];
    const oscs = ratios.map((ratio, i) => {
      const o = ctx.createOscillator();
      o.type = "sawtooth";
      o.frequency.value = t * ratio;
      o.detune.value = i === 2 ? 4 : 0;
      const g = ctx.createGain(); g.gain.value = i === 3 ? 0.5 : 0.3;
      o.connect(g).connect(lp);
      o.start();
      return { o, ratio };
    });
    state.drone = { ctx, oscs };
    els.saPlay.setAttribute("aria-pressed", "true");
  }

  function updateDrone() {
    const t = currentTonic();
    if (!state.drone || !t) return;
    for (const { o, ratio } of state.drone.oscs) o.frequency.setTargetAtTime(t * ratio, state.drone.ctx.currentTime, 0.05);
  }

  function stopDrone() {
    if (!state.drone) return;
    state.drone.ctx.close();
    state.drone = null;
    els.saPlay.setAttribute("aria-pressed", "false");
  }

  // ------------------------------------------------------------------ wiring

  els.listenBtn.addEventListener("click", () => {
    if (state.rec) stopRecording();
    else if (!state.inFlight) startRecording();
  });

  els.fileInput.addEventListener("change", async () => {
    const file = els.fileInput.files[0];
    if (!file) return;
    if (state.rec) await stopRecording();
    state.tonicOverride = null;
    state.lastResult = null;
    clearResults();
    showError("");
    els.progress.style.width = "0";
    await analyse(file, file.name);
    els.fileInput.value = "";
  });

  els.saNote.addEventListener("change", () => {
    state.tonicOverride = midiToHz(Number(els.saNote.value));
    reanalyseWithSa();
  });
  els.saDown.addEventListener("click", () => shiftSa(-1));
  els.saUp.addEventListener("click", () => shiftSa(+1));
  els.saPlay.addEventListener("click", () => (state.drone ? stopDrone() : startDrone()));
  els.saReset.addEventListener("click", () => {
    state.tonicOverride = null;
    els.saReset.hidden = true;
    if (state.rec) state.rec.nextSendAt = 0;
    else if (state.lastAudio) analyse(state.lastAudio, state.lastName);
  });

  buildSaPicker();
  fetch("ragas").then((r) => r.ok ? r.json() : null).then((ragas) => {
    if (!ragas) return;
    const names = Object.values(ragas).map((x) => x.display).sort();
    els.ragaCount.textContent = names.length;
    els.ragaList.textContent = names.join(", ");
  }).catch(() => {});
})();
