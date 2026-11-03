const labels = {
  " Age (yrs)":"Age (years)","Weight (Kg)":"Weight (kg)","Height(Cm) ":"Height (cm)","BMI":"BMI",
  "Blood Group":"Blood group","Pulse rate(bpm) ":"Pulse rate (bpm)","RR (breaths/min)":"Respiratory rate (breaths/min)","Hb(g/dl)":"Hemoglobin (g/dL)",
  "Cycle(R/I)":"Cycle category","Cycle length(days)":"Cycle length (days)","Marraige Status (Yrs)":"Marriage duration (years)","Pregnant(Y/N)":"Pregnancy category","No. of abortions":"Number of abortions",
  "  I   beta-HCG(mIU/mL)":"I beta-HCG","II    beta-HCG(mIU/mL)":"II beta-HCG","FSH(mIU/mL)":"FSH","LH(mIU/mL)":"LH","FSH/LH":"FSH/LH","Hip(inch)":"Hip","Waist(inch)":"Waist","Waist:Hip Ratio":"Waist-to-hip ratio","TSH (mIU/L)":"TSH","AMH(ng/mL)":"AMH","PRL(ng/mL)":"PRL","Vit D3 (ng/mL)":"Vitamin D3","PRG(ng/mL)":"PRG","RBS(mg/dl)":"RBS","BP _Systolic (mmHg)":"Systolic blood pressure","BP _Diastolic (mmHg)":"Diastolic blood pressure",
  "Weight gain(Y/N)":"Weight gain","hair growth(Y/N)":"Hair growth","Skin darkening (Y/N)":"Skin darkening","Hair loss(Y/N)":"Hair loss","Pimples(Y/N)":"Pimples","Fast food (Y/N)":"Fast food response","Reg.Exercise(Y/N)":"Regular exercise","Follicle No. (L)":"Left follicle count","Follicle No. (R)":"Right follicle count","Avg. F size (L) (mm)":"Average left follicle size (mm)","Avg. F size (R) (mm)":"Average right follicle size (mm)","Endometrium (mm)":"Endometrium (mm)"
};
const sections = [
  {title:"Patient information & measurements", keys:[" Age (yrs)","Weight (Kg)","Height(Cm) ","BMI","Blood Group","Pulse rate(bpm) ","RR (breaths/min)","Hb(g/dl)"]},
  {title:"Reproductive & menstrual information", keys:["Cycle(R/I)","Cycle length(days)","Marraige Status (Yrs)","Pregnant(Y/N)","No. of abortions"]},
  {title:"Laboratory measurements", keys:["  I   beta-HCG(mIU/mL)","II    beta-HCG(mIU/mL)","FSH(mIU/mL)","LH(mIU/mL)","FSH/LH","TSH (mIU/L)","AMH(ng/mL)","PRL(ng/mL)","Vit D3 (ng/mL)","PRG(ng/mL)","RBS(mg/dl)"]},
  {title:"Symptoms & lifestyle", keys:["Weight gain(Y/N)","hair growth(Y/N)","Skin darkening (Y/N)","Hair loss(Y/N)","Pimples(Y/N)","Fast food (Y/N)","Reg.Exercise(Y/N)"]},
  {title:"Measurements & ultrasound fields", keys:["Hip(inch)","Waist(inch)","Waist:Hip Ratio","BP _Systolic (mmHg)","BP _Diastolic (mmHg)","Follicle No. (L)","Follicle No. (R)","Avg. F size (L) (mm)","Avg. F size (R) (mm)","Endometrium (mm)"]}
];
const $ = (selector) => document.querySelector(selector);
const numericKeys = new Set();
let schema;

function createField(key) {
  const wrap = document.createElement("div"); wrap.className = "field";
  const id = `feature-${schema.numerical.includes(key) ? "n" : "c"}-${schema.numerical.concat(schema.categorical).indexOf(key)}`;
  const label = document.createElement("label"); label.htmlFor = id; label.textContent = labels[key] || key.trim();
  let control;
  if (schema.numerical.includes(key)) {
    numericKeys.add(key); control = document.createElement("input"); control.type = "number"; control.step = "any"; control.inputMode = "decimal";
    control.addEventListener("input", () => { control.setCustomValidity(control.value && !Number.isFinite(Number(control.value)) ? "Enter a finite number." : ""); });
  } else {
    control = document.createElement("select"); const blank = document.createElement("option"); blank.value = ""; blank.textContent = "Choose a recorded category"; control.append(blank);
    for (const value of schema.categories[key]) { const option = document.createElement("option"); option.value = value; option.textContent = value; control.append(option); }
  }
  control.id = id; control.name = key; control.autocomplete = "off";
  const help = document.createElement("small"); help.textContent = "Optional; blank uses the existing fitted training imputer.";
  wrap.append(label, control, help); return wrap;
}

function renderForm() {
  const target = $("#form-sections");
  for (const section of sections) {
    const box = document.createElement("section"); box.className = "form-section";
    const heading = document.createElement("h3"); heading.textContent = section.title;
    const grid = document.createElement("div"); grid.className = "field-grid";
    for (const key of section.keys) { if (schema.numerical.includes(key) || schema.categorical.includes(key)) grid.append(createField(key)); }
    box.append(heading, grid); target.append(box);
  }
  const rendered = [...document.querySelectorAll("#prediction-form [name]")].map((x) => x.name);
  if (rendered.length !== schema.numerical.length + schema.categorical.length || new Set(rendered).size !== rendered.length) throw new Error("The rendered form does not match the fitted model input schema.");
  $("#threshold-info").textContent = Number(schema.threshold).toFixed(2);
  $("#predict-button").disabled = false;
}

async function initialize() {
  try { const response = await fetch("/schema"); if (!response.ok) throw new Error("Could not load model input schema."); schema = await response.json(); renderForm(); }
  catch (error) { $("#load-error").hidden = false; $("#load-error").textContent = `Dashboard could not load the fitted model artifacts: ${error.message}`; }
}

$("#prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const errors = $("#validation-errors"); errors.hidden = true;
  const payload = {}; const controls = [...document.querySelectorAll("#prediction-form [name]")];
  for (const control of controls) payload[control.name] = control.value;
  for (const key of numericKeys) { const value = payload[key]; if (value !== "" && !Number.isFinite(Number(value))) { errors.hidden = false; errors.textContent = `${labels[key] || key}: enter a valid finite number.`; return; } }
  $("#predict-button").disabled = true; $("#predict-button").textContent = "Calculating…";
  try {
    const response = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)}); const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction request failed.");
    const positive = result.predicted_label === 1; const card = $("#result"); card.hidden = false; card.className = `result-card ${positive ? "positive" : "negative"}`;
    card.innerHTML = `<p class="eyebrow">MODEL OUTPUT · FIXED THRESHOLD ${Number(result.threshold).toFixed(2)}</p><h2>Prediction: PCOS ${positive ? "Positive" : "Negative"}</h2><div class="probability">${(result.probability * 100).toFixed(2)}%</div><p>Predicted probability for the model's positive class.</p>`;
    card.scrollIntoView({behavior:"smooth",block:"center"});
  } catch (error) { errors.hidden = false; errors.textContent = error.message; }
  finally { $("#predict-button").disabled = false; $("#predict-button").innerHTML = 'Predict PCOS <span aria-hidden="true">→</span>'; }
});

initialize();
