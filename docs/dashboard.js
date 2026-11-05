/* Read-only dashboard; all metric values originate in verified report data. */
'use strict';
let evidence;
const byId = id => document.getElementById(id);
const fixed = value => Number(value).toFixed(4);
const percent = value => (100 * Number(value)).toFixed(1) + '%';

/** Populate table cells through safe text nodes. */
function populateTable(id, rows, bestIndex = -1) {
  const body = byId(id).querySelector('tbody'); body.replaceChildren();
  rows.forEach((values, index) => {
    const tr = document.createElement('tr');
    if (index === bestIndex) tr.classList.add('best');
    values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
    body.append(tr);
  });
}
/** Display one training-OOF row without using holdout labels. */
function showThreshold() {
  const model = evidence.models[byId('model').value];
  const row = model.threshold_grid[Number(byId('threshold').value)];
  byId('threshold-value').textContent = row.threshold.toFixed(2);
  byId('threshold-result').textContent = `Training OOF development: precision ${percent(row.precision)} · recall ${percent(row.recall)} · F1 ${fixed(row.f1)}. Frozen selected threshold: ${model.threshold.toFixed(2)}.`;
}
/** Change model-specific figures and reset the OOF slider to its selected value. */
function showModel() {
  const name = byId('model').value, model = evidence.models[name], cm = model.confusion_matrix;
  byId('confusion').src = `milestone3/confusion_${name}.png`;
  byId('importance').src = `milestone3/importance_${name}.png`;
  byId('threshold-chart').src = `milestone3/threshold_${name}.png`;
  byId('confusion-caption').textContent = `${model.label}: threshold ${model.threshold.toFixed(2)}. ${cm[1][1]} responders reached; ${cm[0][1]} nonresponders contacted; ${cm[1][0]} responders missed; ${cm[0][0]} nonresponders correctly excluded.`;
  byId('importance-caption').textContent = name === 'logistic_regression' ? 'Signed coefficients: one training SD for numeric predictors; category indicators are unscaled. Conditional associations, not causal effects.' : 'Normalized impurity importance. Related predictors can share importance; values are not causal effects.';
  byId('threshold').value = model.threshold_grid.findIndex(row => Math.abs(row.threshold-model.threshold)<1e-8);
  populateTable('permutation', [...model.permutation].sort((a,b)=>b.importance-a.importance).slice(0,10).map(row=>[row.feature,fixed(row.importance),fixed(row.std)]));
  showThreshold();
}
/** Render verified metrics, profiles and marketing hypotheses. */
function render(data) {
  if (!data.overview || !Array.isArray(data.comparison) || data.comparison.length!==4 || !data.models || !Array.isArray(data.clusters?.profiles)) throw new Error('Invalid report bundle');
  const fields=['threshold','cv_f1','cv_f1_sd','test_accuracy','test_precision','test_recall','test_f1','test_roc_auc','test_average_precision'];
  data.comparison.forEach(row=>{if (!fields.every(key=>Number.isFinite(row[key])) || !data.models[row.model]?.threshold_grid?.length) throw new Error('Invalid model evidence');});
  evidence=data; const o=data.overview;
  [['Customers',o.rows.toLocaleString()],['Responders',`${o.responders} · ${percent(o.prevalence)}`],['Nonresponders',o.nonresponders.toLocaleString()],['Primary KPI','Positive F1']].forEach(([label,value])=>{
    const card=document.createElement('div'); card.className='card'; const strong=document.createElement('strong'); strong.textContent=value; card.append(label,strong); byId('overview-cards').append(card);
  });
  const best=data.comparison.find(row=>row.model===data.best_test_f1), cv=data.comparison.find(row=>row.model===data.best_cv_f1);
  byId('design').textContent=`${o.train_rows} training customers · ${o.test_rows} held-out customers · ${o.cv_folds}-fold stratified CV · seed 42. Test response prevalence: ${percent(o.test_prevalence)}.`;
  byId('winner').textContent=`${best.label} has the highest observed test F1 (${fixed(best.test_f1)}). ${cv.label} remains the training-CV benchmark (${fixed(cv.cv_f1)}). Small differences do not establish a statistically reliable winner.`;
  populateTable('comparison',data.comparison.map(row=>[row.label,row.threshold.toFixed(2),`${fixed(row.cv_f1)} ± ${fixed(row.cv_f1_sd)}`,...['test_accuracy','test_precision','test_recall','test_f1','test_roc_auc','test_average_precision'].map(key=>fixed(row[key]))]),data.comparison.findIndex(row=>row.model===data.best_test_f1));
  data.comparison.forEach(row=>{const option=document.createElement('option');option.value=row.model;option.textContent=row.label;byId('model').append(option);});
  byId('model').value=data.best_test_f1;
  const clusters=data.clusters;
  byId('cluster-design').textContent=`K=${clusters.k}, silhouette ${fixed(clusters.silhouette)}. Selected from K=2–8 by highest silhouette with every cluster ≥5%. All ${o.rows} customers included.`;
  populateTable('profiles',clusters.profiles.map(row=>[`Cluster ${row.cluster}`,row.size,percent(row.share),Math.round(row.Income).toLocaleString(),row.Total_Spending.toFixed(1),row.NumWebPurchases.toFixed(2),row.NumStorePurchases.toFixed(2),row.Kidhome.toFixed(2),`${percent(row.response_rate)} (${row.responders}/${row.size})`]));
  const ordered=[...clusters.profiles].sort((a,b)=>a.Total_Spending-b.Total_Spending),low=ordered[0],high=ordered[ordered.length-1];
  const suggestions=[
    ['Test offers with the higher-spending group',`Cluster ${high.cluster} averages ${high.Total_Spending.toFixed(0)} in total spending with ${percent(high.response_rate)} response, versus ${percent(low.response_rate)} in Cluster ${low.cluster}. Test complementary or premium offers with randomized controls; improvement is a hypothesis.`],
    ['Match outreach to capacity',`${best.label} achieves ${percent(best.test_precision)} precision at its frozen threshold. Limited capacity can motivate a precision-oriented development threshold; broader outreach may favor recall. Costs and benefits are unmeasured.`],
    ['Check recency and historical engagement','Recency, tenure and campaign history appear in model interpretation. Investigate these signals only after confirming pre-offer availability. Importance does not establish cause or a universal cutoff.'],
    ['Protect the next evaluation','Keep this holdout as historical evidence. Further tuning should use training-only or nested CV and fresh outcomes. Review segment stability before assigning budgets; measure incremental response with randomized campaigns.']];
  suggestions.forEach(([title,text])=>{const article=document.createElement('article'),h=document.createElement('h3'),p=document.createElement('p');h.textContent=title;p.textContent=text;article.append(h,p);byId('business').append(article);});
  byId('provenance').textContent=`Pinned experiments: October 13, 2026. Shared split: ${data.split_id}. Metrics derive from saved predictions and verified fitted artifacts.`;
  byId('evidence').hidden=false;byId('status').textContent='Verified saved experiments · no simulated metrics';showModel();
}
/** Load generated data and show recovery steps for missing or invalid artifacts. */
async function load() {
  try {const response=await fetch('milestone3/dashboard_data.json',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);render(await response.json());}
  catch(error){byId('evidence').hidden=true;byId('overview-cards').replaceChildren();byId('status').className='error';byId('status').textContent=`Results unavailable (${error.message}). Run python -m experiments.milestone3.build, then python -m http.server 8000 --directory docs. See DASHBOARD_GUIDE.md.`;}
}
byId('nav-toggle').addEventListener('click',()=>{const open=byId('navigation').classList.toggle('open');byId('nav-toggle').setAttribute('aria-expanded',String(open));});
document.querySelectorAll('#navigation a').forEach(link=>link.addEventListener('click',()=>{byId('navigation').classList.remove('open');byId('nav-toggle').setAttribute('aria-expanded','false');}));
byId('model').addEventListener('change',showModel);byId('threshold').addEventListener('input',showThreshold);
document.querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{const warning=document.createElement('p');warning.textContent=`Figure unavailable: ${img.getAttribute('src')}. Regenerate report assets.`;img.replaceWith(warning);}));
load();
