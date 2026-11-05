/** Logic-level tests against a minimal DOM double; these do not claim browser rendering QA. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('docs/dashboard.js', 'utf8');
const bundle = JSON.parse(fs.readFileSync('docs/milestone3/dashboard_data.json', 'utf8'));
const html = fs.readFileSync('docs/index.html', 'utf8');

/** Build isolated DOM doubles and run the actual dashboard script with a chosen fetch result. */
async function setup(payload, status = 200) {
  const ids = new Map();
  const element = () => ({
    children: [], textContent: '', value: '', hidden: false, className: '', attributes: {}, handlers: {},
    classList: {add(){}, remove(){}, toggle(){return true;}},
    append(...nodes){this.children.push(...nodes);}, replaceChildren(...nodes){this.children=nodes;},
    addEventListener(name,handler){this.handlers[name]=handler;},
    setAttribute(name,value){this.attributes[name]=value;},
    removeAttribute(name){delete this.attributes[name];},
    hasAttribute(name){return Object.hasOwn(this.attributes,name);},
    contains(target){return target===this || target===this.button;},
    querySelector(){if(!this.body)this.body=element();return this.body;}
  });
  for(const match of html.matchAll(/\bid="([^"]+)"/g))ids.set(match[1],element());
  const info=element(), infoButton=element();
  info.button=infoButton;infoButton.parentElement=info;
  info.querySelector=()=>infoButton;
  const document = {
    handlers:{},addEventListener(name,handler){this.handlers[name]=handler;},
    getElementById(id){return ids.get(id) || null;},
    createElement:element,
    querySelectorAll(selector){
      if(selector==='.info button')return [infoButton];
      if(selector==='.info')return [info];
      if(selector==='.info[data-open]')return info.hasAttribute('data-open')?[info]:[];
      return [];
    }
  };
  const context = vm.createContext({document, console, fetch:async()=>({ok:status===200,status,json:async()=>payload})});
  vm.runInContext(source,context);
  await new Promise(resolve=>setImmediate(resolve));
  return {ids,context,document,info,infoButton};
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
    assert.match(ids.get('threshold-result').textContent,/Precision.*Recall.*F1/);
    assert.match(ids.get('threshold').attributes['aria-valuetext'],/training OOF F1/);
    for(const index of [0,16]){
      ids.get('threshold').value=index;vm.runInContext('showThreshold()',context);
      assert.equal(ids.get('threshold-value').textContent,bundle.models[name].threshold_grid[index].threshold.toFixed(2));
    }
    assert.equal(ids.get('comparison').body.children.length,4);
  }
});

test('information tips support touch pinning, Escape and outside-click dismissal',async()=>{
  const {info,infoButton,document}=await setup(bundle);
  infoButton.handlers.click();
  assert.equal(info.hasAttribute('data-open'),true);
  assert.equal(infoButton.attributes['aria-expanded'],'true');
  document.handlers.keydown({key:'Escape'});
  assert.equal(info.hasAttribute('data-open'),false);
  assert.equal(info.hasAttribute('data-dismissed'),true);
  infoButton.handlers.focus();
  assert.equal(info.hasAttribute('data-dismissed'),false);
  infoButton.handlers.click();
  document.handlers.click({target:{closest:()=>null}});
  assert.equal(info.hasAttribute('data-open'),false);
  assert.equal(info.hasAttribute('data-dismissed'),true);
  assert.equal(infoButton.attributes['aria-expanded'],'false');
});

test('visible controls retain labels and every information tip has an accessible target',()=>{
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
  assert.equal(new Set(ids).size,ids.length);
  for(const target of html.matchAll(/aria-describedby="([^"]+)"/g))assert.ok(ids.includes(target[1]));
  for(const id of ['model','threshold'])assert.match(html,new RegExp(`<label for="${id}">`));
  const buttons=[...html.matchAll(/<button\b([^>]+)>i<\/button>/g)];
  assert.ok(buttons.length>=6);
  for(const [,attributes] of buttons){assert.match(attributes,/aria-label="[^"]+"/);assert.match(attributes,/aria-describedby=/);}
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
