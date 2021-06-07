// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});

console.log(tf.layers.depthwiseConv2d)
//tests done
console.log(true&false?'yes':'no');

//window.GRAPH_URL = "model.json";
//code from distill.pub google colab demo
const sleep = (ms)=>new Promise(resolve => setTimeout(resolve, ms));
  
const parseConsts = model_graph=>{
const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
                'DT_FLOAT':['float32', 'floatVal', Float32Array]};

const consts = {};
model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
  const v = node.attr.value.tensor;
  const [dtype, field, arrayType] = dtypes[v.dtype];
  if (!v.tensorShape.dim) {
    consts[node.name] = [tf.scalar(v[field][0], dtype)];
  } else {
    // if there is a 0-length dimension, the exported graph json lacks "size"
    const shape = v.tensorShape.dim.map(d=>(!d.size) ? 0 : parseInt(d.size));
    let arr;
    if (v.tensorContent) {
      const data = atob(v.tensorContent);
      const buf = new Uint8Array(data.length);
      for (var i=0; i<data.length; ++i) {
        buf[i] = data.charCodeAt(i);
      }
      arr = new arrayType(buf.buffer);
    } else {
      const size = shape.reduce((a, b)=>a*b);
      arr = new arrayType(size);
      if (size) {
        arr.fill(v[field][0]);
      }
    }
    consts[node.name] = [tf.tensor(arr, shape, dtype)];
  }
}));
return consts;
}
  
  const run = async ()=>{
    //Graph_url is set in model.js and defines the tensorflow graph
    const r = await fetch(GRAPH_URL);
    const consts = parseConsts(await r.json());
    
    const model = await tf.loadGraphModel(GRAPH_URL);
    Object.assign(model.weights, consts);
    let mro = [-0.23187873,  0.8748097,   0.21108331, -0.3377599,   1.5307932,  -0.18755093];  
    let lzrd =  [ 1.6492957,   0.77366227,  1.4322221,   0.5890011,  -0.23485255,  0.38630953]
    let elephant= [ 0.11262906, -2.482515,    1.6564426,   0.8290265,  -0.04282238,  0.00299857]
    let whale = [ 1.1224561,  -1.9966718,  -0.7790339,  -0.25473234,  0.45072618,  0.8468036 ]
    let seed = new Array(26).fill(0).map((x, i)=>i>19?mro[i-20]:i<3?0:1);
    seed = tf.tensor(seed, [1, 1, 1, 26]);
    
    const D = 96;
    const initState = tf.tidy(()=>{
      const D2 = D/2;
      const a = seed.pad([[0, 0], [D2-1, D2], [D2-1, D2], [0,0]]);
      return a;
    });
    
    const state = tf.variable(initState);
    const [_, h, w, ch] = state.shape;
    
    const damage = (x, y, r)=>{
      tf.tidy(()=>{
        const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
        const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
        const mask = rx.add(ry).greater(1.0).expandDims(2);
        state.assign(state.mul(mask));
      });
    }
    
    const plantSeed = (x, y)=>{
      let choice = [mro,lzrd,elephant,whale][Math.floor(Math.random()*4)]
      let seed = new Array(26).fill(0).map((x, i)=>i>19?choice[i-20]:i<3?0:1);
      seed = tf.tensor(seed, [1, 1, 1, 26]);
      const x2 = w-x-seed.shape[2];
      const y2 = h-y-seed.shape[1];
      if (x<0 || x2<0 || y2<0 || y2<0)
        return;
      tf.tidy(()=>{
        const a = seed.pad([[0, 0], [y, y2], [x, x2], [0,0]]);
        state.assign(state.add(a));
      });
    }
    
    const scale = 5;
    
    const canvas = document.getElementById('grid');

    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w*scale}px`;
    canvas.style.height = `${h*scale}px`;
    
    canvas.onmousedown = e=>{
      var rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX-rect.left)/scale);
      const y = Math.floor((e.clientY-rect.top)/scale);
        if (e.buttons == 1) {
          if (e.shiftKey) {
            plantSeed(x, y);  
          } else {
            damage(x, y, 8);
          }
        }
    }
    canvas.onmousemove = e=>{
      var rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX-rect.left)/scale);
      const y = Math.floor((e.clientY-rect.top)/scale);
      if (e.buttons == 1 && !e.shiftKey) {
        damage(x, y, 8);
      }
    }

    function step() {
      tf.tidy(()=>{
        state.assign(model.execute(
            {x:state, fire_rate:tf.tensor(0.5),
            angle:tf.tensor(0.0), step_size:tf.tensor(1.0)}, ['Identity']));
      });
    }

    function render() {
      step();

      const imageData = tf.tidy(()=>{
        const rgba = state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
        const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
        const img = tf.tensor(1.0).sub(a).add(rgba).mul(255);
        const rgbaBytes = new Uint8ClampedArray(img.dataSync());
        return new ImageData(rgbaBytes, w, h);
      });
      ctx.putImageData(imageData, 0, 0);

      requestAnimationFrame(render);
    }
    render();
  }
  run();
  
