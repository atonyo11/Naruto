import {
  FilesetResolver,
  GestureRecognizer
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const cameraEl = document.getElementById("camera");
const shurikenFxEl = document.getElementById("shurikenFx");
const statusEl = document.getElementById("status");

let gestureRecognizer = null;
let running = false;
let lastVideoTime = -1;

const effect = createEffectState(shurikenFxEl, 0.95);

function createEffectState(el, scaleMultiplier = 1) {
  return {
    el,
    scaleMultiplier,
    visible: false,
    x: window.innerWidth * 0.5,
    y: window.innerHeight * 0.5,
    scale: 1,
    rotation: 0,
    smoothX: window.innerWidth * 0.5,
    smoothY: window.innerHeight * 0.5,
    smoothScale: 1,
    smoothRotation: 0
  };
}

function setStatus(text) {
  statusEl.textContent = text;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function averagePoints(points) {
  const sum = points.reduce((acc, p) => {
    acc.x += p.x;
    acc.y += p.y;
    return acc;
  }, { x: 0, y: 0 });
  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
}

function palmCenter(landmarks) {
  const pts = [
    landmarks[0],
    landmarks[5],
    landmarks[9],
    landmarks[13],
    landmarks[17]
  ];
  return averagePoints(pts);
}

function estimatePalmScale(landmarks) {
  const palmWidth = distance(landmarks[5], landmarks[17]);
  const wristToMiddle = distance(landmarks[0], landmarks[9]);
  const raw = (palmWidth * 1.45 + wristToMiddle * 1.1) * 0.5;
  return clamp(raw * 14.0, 2.17, 8.0);
}

function isFingerExtended(tip, pip, mcp, wrist) {
  return tip.y < pip.y && pip.y < mcp.y && tip.y < wrist.y;
}

function isOpenPalmHeuristic(landmarks) {
  const wrist = landmarks[0];
  const thumbOpen = Math.abs(landmarks[4].x - landmarks[2].x) > 0.06;
  const indexOpen = isFingerExtended(landmarks[8], landmarks[6], landmarks[5], wrist);
  const middleOpen = isFingerExtended(landmarks[12], landmarks[10], landmarks[9], wrist);
  const ringOpen = isFingerExtended(landmarks[16], landmarks[14], landmarks[13], wrist);
  const pinkyOpen = isFingerExtended(landmarks[20], landmarks[18], landmarks[17], wrist);
  const extendedCount = [indexOpen, middleOpen, ringOpen, pinkyOpen].filter(Boolean).length;
  return extendedCount >= 3 && thumbOpen;
}

function isClosedPalmHeuristic(landmarks) {
  const wrist = landmarks[0];
  const curled =
    landmarks[8].y > landmarks[6].y &&
    landmarks[12].y > landmarks[10].y &&
    landmarks[16].y > landmarks[14].y &&
    landmarks[20].y > landmarks[18].y;

  const thumbClosed = Math.abs(landmarks[4].x - landmarks[2].x) < 0.05;
  const compactPalm =
    distance(landmarks[8], wrist) < distance(landmarks[5], wrist) * 1.15 &&
    distance(landmarks[12], wrist) < distance(landmarks[9], wrist) * 1.15;

  return curled && (thumbClosed || compactPalm);
}

function getGestureName(result, handIndex) {
  return result.gestures?.[handIndex]?.[0]?.categoryName || "";
}

function isOpenGesture(result, handIndex) {
  const gesture = getGestureName(result, handIndex);
  if (gesture === "Open_Palm") return true;
  const landmarks = result.landmarks?.[handIndex];
  if (!landmarks) return false;
  return isOpenPalmHeuristic(landmarks);
}

function isClosedGesture(result, handIndex) {
  const gesture = getGestureName(result, handIndex);
  if (gesture === "Closed_Fist") return true;
  const landmarks = result.landmarks?.[handIndex];
  if (!landmarks) return false;
  return isClosedPalmHeuristic(landmarks);
}

function chooseActiveHand(result) {
  const hands = result.landmarks || [];
  let bestOpen = null;
  let anyClosed = false;

  for (let i = 0; i < hands.length; i++) {
    const landmarks = hands[i];
    const center = palmCenter(landmarks);
    const scale = estimatePalmScale(landmarks);
    const isOpen = isOpenGesture(result, i);
    const isClosed = isClosedGesture(result, i);

    if (isClosed) {
      anyClosed = true;
    }

    if (isOpen) {
      const candidate = { landmarks, center, scale };
      if (!bestOpen || candidate.scale > bestOpen.scale) {
        bestOpen = candidate;
      }
    }
  }

  return {
    hand: bestOpen,
    anyClosed
  };
}

function showEffect(state) {
  if (!state.visible) {
    state.visible = true;
    state.el.currentTime = 0;
    state.el.play().catch(() => {});
  }
  state.el.style.opacity = "1";
}

function hideEffect(state) {
  state.visible = false;
  state.el.style.opacity = "0";
}

function updateEffectTransform(state, targetX, targetY, targetScale, targetRotation = 0) {
  state.x = targetX;
  state.y = targetY;
  state.scale = targetScale;
  state.rotation = targetRotation;

  state.smoothX = lerp(state.smoothX, state.x, 0.32);
  state.smoothY = lerp(state.smoothY, state.y, 0.32);
  state.smoothScale = lerp(state.smoothScale, state.scale, 0.28);
  state.smoothRotation = lerp(state.smoothRotation, state.rotation, 0.18);

  state.el.style.left = state.smoothX + "px";
  state.el.style.top = state.smoothY + "px";
  state.el.style.transform =
    `translate(-50%, -50%) rotate(${state.smoothRotation}deg) scale(${state.smoothScale})`;
}

async function initRecognizer() {
  setStatus("Loading MediaPipe…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );

  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.55
  });

  setStatus("MediaPipe ready.");
}

async function startCamera() {
  setStatus("Requesting camera access…");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  });

  cameraEl.srcObject = stream;

  await new Promise((resolve) => {
    cameraEl.onloadedmetadata = () => resolve();
  });

  await cameraEl.play();

  shurikenFxEl.play().catch(() => {});
  shurikenFxEl.pause();

  setStatus("Camera started.");
}

function renderLoop() {
  if (!running) return;

  if (
    gestureRecognizer &&
    cameraEl.readyState >= 2 &&
    cameraEl.currentTime !== lastVideoTime
  ) {
    lastVideoTime = cameraEl.currentTime;
    const nowMs = performance.now();
    const result = gestureRecognizer.recognizeForVideo(cameraEl, nowMs);

    const { hand, anyClosed } = chooseActiveHand(result);

    if (hand) {
      const x = (1 - hand.center.x) * window.innerWidth;
      const y = hand.center.y * window.innerHeight;
      const scale = hand.scale * effect.scaleMultiplier;

      showEffect(effect);
      updateEffectTransform(effect, x, y, scale, 0);
      setStatus("Open hand detected. Shuriken on.");
    } else if (anyClosed) {
      hideEffect(effect);
      setStatus("Closed hand detected. Shuriken off.");
    } else {
      setStatus(effect.visible ? "Tracking hand..." : "No open hand detected.");
    }
  }

  requestAnimationFrame(renderLoop);
}

async function boot() {
  try {
    await initRecognizer();
    await startCamera();
    running = true;
    renderLoop();
  } catch (err) {
    console.error(err);
    setStatus("Could not start camera or MediaPipe. Please allow webcam access and reload.");
  }
}

window.addEventListener("resize", () => {
  effect.smoothX = window.innerWidth * 0.5;
  effect.smoothY = window.innerHeight * 0.5;
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    hideEffect(effect);
  }
});

boot();
