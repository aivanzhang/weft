'use strict';
const maskData = JSON.parse(document.getElementById('mask-data').textContent);
const frameSlider = document.getElementById('frame');
const maskCanvas = document.getElementById('canvas');
const maskContext = maskCanvas.getContext('2d');
const maskPosition = document.getElementById('position');
const maskStage = document.getElementById('stage');
const previousFrame = document.getElementById('previous');
const nextFrame = document.getElementById('next');
const maskLayers = ['masks', 'boxes', 'centers'].map(id => document.getElementById(id));
frameSlider.max = maskData.frames.length - 1;
[maskCanvas.width, maskCanvas.height] = maskData.preview.dimensions;
let selectedMaskFrame = 0, maskRevision = 0, maskBusy = false, renderedMaskFrame = null, sliderTimer = null, maskPageActive = true;

function loadMaskImage(path) {
  return new Promise((resolve, reject) => {
    const picture = new Image();
    picture.onload = () => resolve(picture);
    picture.onerror = () => reject(new Error('Saved image unavailable'));
    picture.src = path;
  });
}

function drawMaskFrame(value) {
  const {frame, rows, pictures, silhouetteRows} = value;
  maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskContext.drawImage(pictures[0], 0, 0, maskCanvas.width, maskCanvas.height);
  maskContext.save();
  maskContext.scale(maskCanvas.width / maskData.source.width_px, maskCanvas.height / maskData.source.height_px);
  maskContext.imageSmoothingEnabled = false;
  if (maskLayers[0].checked) silhouetteRows.forEach((row, index) => {
    const [x, y, right, bottom] = row.box.box_xyxy_halfopen;
    maskContext.drawImage(pictures[index + 1], x, y, right - x, bottom - y);
  });
  maskContext.lineWidth = 2 * maskData.source.width_px / maskCanvas.width;
  for (const row of rows) {
    maskContext.strokeStyle = maskContext.fillStyle = `rgb(${row.color.join(',')})`;
    if (maskLayers[1].checked && row.box) {
      const [x, y, right, bottom] = row.box.box_xyxy_halfopen;
      maskContext.strokeRect(x, y, right - x, bottom - y);
    }
    const center = row.silhouette?.centroid_px;
    if (maskLayers[2].checked && center) {
      maskContext.beginPath();
      maskContext.arc(center[0], center[1], 3 * maskData.source.width_px / maskCanvas.width, 0, Math.PI * 2);
      maskContext.fill();
    }
  }
  maskContext.restore();
  maskCanvas.hidden = false;
  maskStage.setAttribute('aria-busy', 'false');
  maskPosition.textContent = `Frame ${frame.frame} / ${frameSlider.max} · ${frame.time_s.toFixed(3)} s`;
  const original = rows[0]?.original_source;
  const origin = original ? `Original source frame ${original.frame}. ` : '';
  document.getElementById('summary').textContent = origin + rows.map(row => {
    const label = row.silhouette ? `${row.silhouette.foreground_pixels} mask pixels; ${row.silhouette.components_8_connected} connected regions` : row.mask_status.replaceAll('_', ' ');
    return `${row.object_id}: ${label} (${row.phase.replaceAll('_', ' ')}).`;
  }).join(' ');
  document.getElementById('details').textContent = JSON.stringify(rows, null, 2);
}

async function renderMaskSelection() {
  if (maskBusy || !maskPageActive) return;
  maskBusy = true;
  const revision = maskRevision;
  const frame = maskData.frames[selectedMaskFrame];
  const rows = frame.observation_indices.map(index => maskData.observations[index]);
  const silhouetteRows = rows.filter(row => row.raster_path !== null);
  try {
    const loaded = await Promise.allSettled([frame.preview, ...silhouetteRows.map(row => row.raster_path)].map(loadMaskImage));
    if (loaded.some(item => item.status === 'rejected')) throw new Error('Saved image unavailable');
    const pictures = loaded.map(item => item.value);
    if (revision === maskRevision) {
      renderedMaskFrame = {frame, rows, pictures, silhouetteRows};
      drawMaskFrame(renderedMaskFrame);
    }
  } catch {
    if (revision === maskRevision) {
      maskPosition.textContent = 'Could not load this saved frame. The report and original evidence remain available.';
      maskStage.setAttribute('aria-busy', 'false');
    }
  } finally {
    maskBusy = false;
    // One active image batch and one latest selection prevent an unbounded seek queue.
    if (revision !== maskRevision && maskPageActive) renderMaskSelection();
  }
}

function selectMaskFrame(index) {
  selectedMaskFrame = Math.max(0, Math.min(maskData.frames.length - 1, index));
  frameSlider.value = selectedMaskFrame;
  maskRevision += 1;
  renderedMaskFrame = null;
  maskCanvas.hidden = true;
  maskStage.setAttribute('aria-busy', 'true');
  previousFrame.disabled = selectedMaskFrame === 0;
  nextFrame.disabled = selectedMaskFrame === maskData.frames.length - 1;
  maskPosition.textContent = `Loading frame ${selectedMaskFrame}…`;
  renderMaskSelection();
}

frameSlider.addEventListener('input', () => {
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => selectMaskFrame(Number(frameSlider.value)), 60);
});
previousFrame.onclick = () => {clearTimeout(sliderTimer); selectMaskFrame(selectedMaskFrame - 1);};
nextFrame.onclick = () => {clearTimeout(sliderTimer); selectMaskFrame(selectedMaskFrame + 1);};
maskLayers.forEach(layer => layer.addEventListener('change', () => {
  if (renderedMaskFrame) drawMaskFrame(renderedMaskFrame);
}));
window.addEventListener('pagehide', () => {clearTimeout(sliderTimer); maskPageActive = false; maskRevision += 1;});
window.addEventListener('pageshow', () => {if (!maskPageActive) {maskPageActive = true; selectMaskFrame(selectedMaskFrame);}});
selectMaskFrame(0);
