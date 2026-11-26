// Run with node --test tests/test_phone_viewport.mjs (no npm dependencies).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/visualization/mobile/index.html', import.meta.url), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function setup(options = {}) {
  const elements = {};
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) {
    const classes = new Set();
    const captures = new Set();
    elements[id] = Object.assign(new EventTarget(), {
      style: { setProperty(key, value) { this[key] = value; } },
      attributes: {}, captures, hidden: true, srcObject: null,
      value: id === 'mode' ? 'human' : 'lamborghini',
      classList: {
        add(value) { classes.add(value); },
        remove(value) { classes.delete(value); },
        contains(value) { return classes.has(value); },
        toggle(value, enabled) { enabled ? classes.add(value) : classes.delete(value); },
      },
      setAttribute(key, value) { this.attributes[key] = value; },
      setPointerCapture(id) { captures.add(id); },
      hasPointerCapture(id) { return captures.has(id); },
      releasePointerCapture(id) { captures.delete(id); },
      play() { this.playCount = (this.playCount || 0) + 1; return Promise.resolve(); },
    });
  }
  const visualViewport = Object.assign(new EventTarget(), { width: 390, height: 664, offsetLeft: 0, offsetTop: 0 });
  const window = Object.assign(new EventTarget(), { innerWidth: 390, innerHeight: 844, visualViewport });
  let fullscreenCalls = 0, peerCreations = 0, now = 1000;
  const document = Object.assign(new EventTarget(), {
    hidden: false,
    getElementById: id => elements[id],
    documentElement: { requestFullscreen() { fullscreenCalls++; return Promise.resolve(); } },
  });
  const frames = [], timers = new Map(), messages = [];
  let nextId = 1;
  const context = vm.createContext({
    window, document, screen: { orientation: new EventTarget() },
    navigator: {}, location: { hash: options.token ? '#token=test' : '' }, URLSearchParams, console, AbortController,
    MediaStream: class { constructor(tracks) {this.tracks=tracks;} },
    performance: { now: () => now },
    requestAnimationFrame(callback) { frames.push(callback); return nextId++; },
    setTimeout(callback,delay) { const id = nextId++; timers.set(id, {callback,delay}); return id; },
    clearTimeout(id) { timers.delete(id); }, setInterval() {},
    RTCPeerConnection: class extends EventTarget {
      constructor() {super();peerCreations++;this.connectionState='new';this.iceGatheringState='complete';this.channels=[];}
      createDataChannel(label) {const channel={label,readyState:'connecting',bufferedAmount:0,
        send(value){messages.push(JSON.parse(value));},close(){this.readyState='closed';this.onclose?.();}};this.channels.push(channel);return channel;}
      addTransceiver() {}
      async createOffer(){return {type:'offer',sdp:'synthetic'};}
      async setLocalDescription(value){this.localDescription=value;}
      async setRemoteDescription(){if(options.openBeforeAnswer){this.connectionState='connected';this.channels.forEach(c=>c.readyState='open');this.onconnectionstatechange?.();}}
      close(){this.connectionState='closed';this.onconnectionstatechange?.();this.channels.forEach(c=>c.close());}
    },
    fetch:options.fetch|| (async()=>({ok:true,json:async()=>({type:'answer',sdp:'synthetic'})})),
    captureInput(message) { messages.push(JSON.parse(message)); },
  });
  new vm.Script(script).runInContext(context);
  const run = source => vm.runInContext(source, context);
  const flush = () => { while (frames.length) frames.shift()(); };
  const rotate = (width, height) => {
    visualViewport.width = width; visualViewport.height = height;
    window.dispatchEvent(new Event('orientationchange'));
    flush();
  };
  return { elements, window, document, visualViewport, run, messages, flush, rotate,
    advance(ms) { now += ms; }, fullscreenCalls: () => fullscreenCalls, peerCreations: () => peerCreations,
    fireTimer(delay) {const entry=[...timers].find(([,item])=>item.delay===delay);assert.ok(entry,'Expected timer');timers.delete(entry[0]);return entry[1].callback();} };
}

test('uses visible Safari height instead of the taller layout viewport', () => {
  const s = setup();
  assert.equal(s.elements.app.style['--app-height'], '664px');
  assert.equal(s.elements.app.classList.contains('portrait'), true);
  s.rotate(844, 290);
  assert.equal(s.elements.app.style['--app-width'], '844px');
  assert.equal(s.elements.app.style['--app-height'], '290px');
  assert.equal(s.elements.app.classList.contains('portrait'), false);
  assert.equal(s.elements.app.classList.contains('tiny-landscape'), true);
  s.rotate(390, 664);
  assert.equal(s.elements.app.classList.contains('portrait'), true);
  assert.equal(s.elements.app.classList.contains('compact-landscape'), false);
});

test('rotation releases held input and sends brake without rebuilding WebRTC', () => {
  const s = setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};raceState='running';gas=true;steering=.8;activePointers.set('gas',7);$('gas').setPointerCapture(7);$('gas').classList.add('held');sendInput();");
  assert.equal(s.messages.at(-1).throttle, 1);
  s.rotate(844, 320);
  assert.equal(s.messages.at(-1).throttle, -1);
  assert.equal(s.messages.at(-1).steering, 0);
  assert.equal(s.elements.gas.captures.size, 0);
  assert.equal(s.elements.gas.classList.contains('held'), false);
  assert.equal(s.elements.steerLeft.attributes['aria-pressed'], 'false');
  assert.equal(s.elements.steerRight.attributes['aria-pressed'], 'false');
  s.advance(400); s.run('sendInput()');
  assert.equal(s.messages.at(-1).throttle, 0);
  assert.equal(s.peerCreations(), 0);
});

test('an old cancelled touch cannot release a new gas press after rotation', () => {
  const s = setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};raceState='running';");
  s.rotate(844, 320); s.advance(400);
  s.elements.gas.onpointerdown({ pointerId: 8, preventDefault() {} });
  s.elements.gas.onlostpointercapture({ pointerId: 7, preventDefault() {} });
  s.run('sendInput()');
  assert.equal(s.messages.at(-1).throttle, 1);
  s.elements.gas.onpointerup({ pointerId: 8, preventDefault() {} });
  assert.equal(s.messages.at(-1).throttle, 0);
});

test('starting a race does not force iPhone video into fullscreen', () => {
  const s = setup();
  s.elements.start.onclick();
  assert.equal(s.fullscreenCalls(), 0);
});

test('video resumes on the same stream, with a tap fallback if Safari blocks play', async () => {
  const s = setup(), stream = {};
  s.elements.view.srcObject = stream;
  s.rotate(844, 320);
  await Promise.resolve();
  assert.equal(s.elements.view.srcObject, stream);
  assert.equal(s.elements.view.playsInline, true);
  assert.ok(s.elements.view.playCount > 0);
  s.elements.view.play = () => Promise.reject(new Error('gesture required'));
  s.run('resumeVideo()');
  await Promise.resolve(); await Promise.resolve();
  assert.equal(s.elements.resumeVideo.hidden, false);
});

test('falls back to window dimensions when VisualViewport is unavailable', () => {
  const s = setup();
  s.window.visualViewport = undefined;
  s.window.innerWidth = 667; s.window.innerHeight = 290;
  s.window.dispatchEvent(new Event('resize')); s.flush();
  assert.equal(s.elements.app.style['--app-width'], '667px');
  assert.equal(s.elements.app.style['--app-height'], '290px');
});

const touch = pointerId => ({ pointerId, preventDefault() {} });

test('direction buttons steer while held, release to centre and work with gas/brake', () => {
  const s = setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};raceState='running';");
  s.elements.steerLeft.onpointerdown(touch(10));
  s.elements.gas.onpointerdown(touch(11));
  assert.equal(s.messages.at(-1).steering, -1);
  assert.equal(s.messages.at(-1).throttle, 1);
  s.elements.brake.onpointerdown(touch(12));
  assert.equal(s.messages.at(-1).throttle, -1);
  s.elements.steerRight.onpointerdown(touch(13));
  assert.equal(s.messages.at(-1).steering, 0);
  s.elements.steerLeft.onpointerup(touch(10));
  assert.equal(s.messages.at(-1).steering, 1);
  s.elements.steerRight.onpointercancel(touch(13));
  assert.equal(s.messages.at(-1).steering, 0);
});

test('stale pointer release, blur and orientation cannot leave steering held', () => {
  const s = setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};raceState='running';");
  s.elements.steerLeft.onpointerdown(touch(20));
  s.elements.steerLeft.onlostpointercapture(touch(19));
  assert.equal(s.messages.at(-1).steering, -1);
  s.window.dispatchEvent(new Event('blur'));
  assert.equal(s.messages.at(-1).steering, 0);
  s.elements.steerRight.onpointerdown(touch(21));
  s.rotate(844, 320);
  assert.equal(s.messages.at(-1).steering, 0);
  assert.equal(s.elements.steerRight.captures.size, 0);
});

test('elimination shows its reason, disables driving and recovery, and has no rank', () => {
  const s = setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};telemetry({state:'running',mode:'human',finished:true,result_label:'Bị loại: đè vạch liền',clean_success:false,recover_allowed:false});");
  assert.equal(s.elements.banner.textContent, 'Bị loại: đè vạch liền');
  assert.equal(s.elements.recover.hidden, true);
  assert.equal(s.elements.steerLeft.disabled, true);
  assert.ok(s.elements.raceInfo.textContent.includes('Không xếp hạng'));
  s.elements.gas.onpointerdown(touch(30));
  s.run('sendInput()');
  assert.equal(s.messages.at(-1).throttle, -1);
  s.run("telemetry({state:'running',mode:'practice',recover_allowed:true});");
  assert.equal(s.elements.recover.hidden, false);
  assert.ok(s.elements.ruleSummary.textContent.includes('không xếp hạng'));
});

test('connection waits for both channels and cannot regress after answer completes', async () => {
  const s=setup({token:true});await s.run('connect()');
  assert.equal(s.run('connected'),false);
  s.run("pc.connectionState='connected';commands.readyState='open';commands.onopen();");
  assert.equal(s.run('connected'),false);
  s.run("drive.readyState='open';drive.onopen();");
  assert.equal(s.run('connected'),true);
  assert.equal(s.elements.connection.textContent,'Đã kết nối tay lái');
  const fast=setup({token:true,openBeforeAnswer:true});await fast.run('connect()');
  assert.equal(fast.run('connected'),true);
  assert.equal(fast.elements.connection.textContent,'Đã kết nối tay lái');
  assert.equal(fast.elements.start.disabled,false);
  assert.equal(fast.elements.lobbyDisconnect.hidden,false);
});

test('hung signaling times out, releases the peer and offers retry', async () => {
  const s=setup({token:true,fetch:(_url,options)=>new Promise((resolve,reject)=>{
    options.signal.addEventListener('abort',()=>reject(Object.assign(Error('aborted'),{name:'AbortError'})));
  })});
  const pending=s.run('connect()');
  for(let i=0;i<10;i++)await Promise.resolve();
  await s.fireTimer(15000);await pending;
  assert.equal(s.run('connected'),false);
  assert.equal(s.run('pc'),null);
  assert.equal(s.elements.connect.disabled,false);
  assert.ok(s.elements.error.textContent.includes('15 giây'));
});

test('callbacks from an old peer do not overwrite a successful retry', async () => {
  const s=setup({token:true,openBeforeAnswer:true});await s.run('connect()');
  s.run('oldPeer=pc;oldControl=commands;');await s.run('connect()');
  s.run("oldControl.onopen();oldControl.onclose();oldPeer.ontrack({track:'old'});oldPeer.connectionState='failed';oldPeer.onconnectionstatechange();");
  assert.equal(s.run('connected'),true);
  assert.equal(s.run('pc===oldPeer'),false);
  assert.equal(s.elements.view.srcObject,null);
  assert.equal(s.elements.connection.textContent,'Đã kết nối tay lái');
});

test('joining an AI race is explicit and never shows unusable driving controls', async () => {
  const s=setup({token:true,openBeforeAnswer:true});await s.run('connect()');
  s.run("telemetry({state:'running',mode:'compare',connected:true});");
  assert.equal(s.elements.lobby.style.display,'flex');
  assert.ok(s.elements.start.textContent.includes('VỀ GARAGE'));
  s.elements.start.onclick();
  assert.equal(s.messages.at(-1).command,'join_driver');
  s.run("telemetry({state:'garage',mode:'human',connected:true});");
  assert.ok(s.elements.start.textContent.includes('XUẤT PHÁT'));
});

test('lobby disconnect releases both channels and exposes connect again', async () => {
  const s=setup({token:true,openBeforeAnswer:true});await s.run('connect()');
  assert.equal(s.elements.lobbyDisconnect.hidden,false);
  await s.elements.lobbyDisconnect.onclick();
  assert.equal(s.run('connected'),false);
  assert.equal(s.elements.lobbyDisconnect.hidden,true);
  assert.equal(s.elements.connect.disabled,false);
  assert.equal(s.elements.connect.style.display,'inline-block');
});

test('phone shows the shared race speed limit from telemetry', () => {
  const s=setup();
  s.run("telemetry({state:'running',mode:'human',speed_kmh:19.8,speed_limit_kmh:20});");
  assert.equal(s.elements.speed.textContent,20);
  assert.ok(s.elements.ruleStatus.textContent.includes('Tối đa 20 km/h'));
});

test('opening options releases held gas and blocks driving until closed', () => {
  const s=setup();
  s.run("drive={readyState:'open',bufferedAmount:0,send:captureInput};raceState='running';");
  s.elements.gas.onpointerdown(touch(40));
  assert.equal(s.messages.at(-1).throttle,1);
  s.elements.more.onclick();
  assert.equal(s.elements.driveMenu.hidden,false);
  assert.equal(s.elements.more.attributes['aria-expanded'],'true');
  assert.equal(s.messages.at(-1).throttle,-1);
  s.elements.gas.onpointerdown(touch(41));
  assert.equal(s.messages.at(-1).throttle,-1);
  s.elements.menuClose.onclick();
  assert.equal(s.elements.driveMenu.hidden,true);
  assert.equal(s.messages.at(-1).throttle,0);
});

test('auto upgrades a stable stream to 720p and steps down before sacrificing resolution further', () => {
  const s=setup();s.advance(12000);
  s.run("commands={readyState:'open',send:captureInput};raceState='running';decodedFps=29;networkRTT=12;lastStream={resolution:[960,540],encode_p95_ms:15};");
  s.run('for(let i=0;i<6;i++)adaptStreamQuality()');
  assert.equal(s.messages.at(-1).value,'hd');
  assert.equal(s.run('qualityChoice'),'auto');
  s.advance(9000);
  s.run('lastStream.resolution=[1280,720];decodedFps=18;for(let i=0;i<3;i++)adaptStreamQuality()');
  assert.equal(s.messages.at(-1).value,'balanced');
  s.advance(9000);
  s.run('lastStream.resolution=[960,540];for(let i=0;i<3;i++)adaptStreamQuality()');
  assert.equal(s.messages.at(-1).value,'smooth');
});

test('manual HD stays selected and the lobby and driving selectors remain in sync', () => {
  const s=setup();s.advance(12000);
  s.run("commands={readyState:'open',send:captureInput};raceState='running';");
  s.elements.driveQuality.value='hd';s.elements.driveQuality.onchange();
  assert.equal(s.elements.quality.value,'hd');
  assert.equal(s.messages.at(-1).value,'hd');
  const count=s.messages.length;s.advance(20000);
  s.run('decodedFps=10;networkRTT=200;lastStream={resolution:[1280,720]};for(let i=0;i<10;i++)adaptStreamQuality()');
  assert.equal(s.messages.length,count);
});

test('connecting preserves the desktop car until the player explicitly starts', async () => {
  const s=setup({token:true,openBeforeAnswer:true});await s.run('connect()');
  assert.equal(s.messages.some(m=>m.command==='car_model'||m.command==='mode'),false);
  s.run("telemetry({state:'garage',mode:'human',model:'tesla',model_name:'Tesla Cybertruck'});");
  assert.equal(s.elements.model.value,'tesla');
  s.elements.start.onclick();
  assert.ok(s.messages.some(m=>m.command==='car_model'&&m.value==='tesla'));
});
