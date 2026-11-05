/** Logic-level tests against a minimal DOM double; these do not claim browser rendering QA. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/dashboard.js', 'utf8');
const bundle = JSON.parse(fs.readFileSync('docs/milestone3/dashboard_data.json', 'utf8'));

/** Build isolated DOM doubles and run the actual dashboard script with a chosen fetch result. */
async function setup(payload, status = 200) {
  const ids = new Map();
  const element = () => ({
    children: [], textContent: '', value: '', hidden: false, className: '', attributes: {}, handlers: {},
    classList: {add(){}, remove(){}, toggle(){return true;}},
    append(...nodes){this.children.push(...nodes);}, replaceChildren(...nodes){this.children=nodes;},
    addEventListener(name,handler){this.handlers[name]=handler;},
    setAttribute(name,value){this.attributes[name]=value;},
    querySelector(){if(!this.body)this.body=element();return this.body;}
  });
  const document = {
    getElementById(id){if(!ids.has(id))ids.set(id,element());return ids.get(id);},
    createElement:element, querySelectorAll(){return [];}
  };
  const context = vm.createContext({document, console, fetch:async()=>({ok:status===200,status,json:async()=>payload})});
  vm.runInContext(source,context);
  await new Promise(resolve=>setImmediate(resolve));
  return {ids,context};
}

test('actual exported values populate overview, four rows and two segments',async()=>{
  const {ids}=await setup(bundle);
  assert.equal(ids.get('evidence').hidden,false);
  assert.equal(ids.get('comparison').body.children.length,4);
  assert.equal(ids.get('profiles').body.children.length,2);
  assert.equal(ids.get('overview-cards').children.length,4);
  assert.match(ids.get('winner').textContent,/Random Forest.*0\.5874/);
});

test('all model selections and OOF grid endpoints update without changing test evidence',async()=>{
  const {ids,context}=await setup(bundle);
  for(const name of Object.keys(bundle.models)){
    ids.get('model').value=name; vm.runInContext('showModel()',context);
    assert.equal(ids.get('confusion').src,`milestone3/confusion_${name}.png`);
    assert.match(ids.get('threshold-result').textContent,/Training OOF development/);
    for(const index of [0,16]){
      ids.get('threshold').value=index;vm.runInContext('showThreshold()',context);
      assert.equal(ids.get('threshold-value').textContent,bundle.models[name].threshold_grid[index].threshold.toFixed(2));
    }
    assert.equal(ids.get('comparison').body.children.length,4);
  }
});

test('missing data produces recovery instructions and hides result panels',async()=>{
  const {ids}=await setup(null,404);
  assert.equal(ids.get('evidence').hidden,true);
  assert.equal(ids.get('status').className,'error');
  assert.match(ids.get('status').textContent,/HTTP 404.*experiments\.milestone3\.build/);
});

test('malformed metrics are rejected rather than displayed',async()=>{
  const invalid=structuredClone(bundle);invalid.comparison[0].test_f1=null;
  const {ids}=await setup(invalid);
  assert.equal(ids.get('evidence').hidden,true);
  assert.match(ids.get('status').textContent,/Invalid model evidence/);
});
