
//stole this recursive thing but gonna make not recursive
function zeros(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; i++) {
        var arr2= [];
        for (var j=0; j<dimensions[1];j++){
            arr2.push(0);
        }
        array.push(arr2);
    }

    return array;
}

function random_ones(dim,arr,N){
 for(var i=0;i<N;){ 
    x=Math.floor(Math.random() * dim[0]);
    y=Math.floor(Math.random() * dim[1]);
    if (arr[x][y]==0){
        arr[x][y]=1
        i++;
    }
 }
}

const gpu = new GPU();

const SIZE = 128;




//gpu bound ising proposal using a checkorboard update rule since no neighbours lie on the
//same checkerboard
const Propose = gpu.createKernel(function(grid,bonds,JB,mew,parity,nspins,size) {
    let i=this.thread.y
    let j=this.thread.x
    let s=grid[i][j];
    //This updates grid cells at least a little bit stochastically, the choice of 0.5 is arbitrary
    //Note: if this is set to 1 you will get weirdness, probably due to the sketchy math.random function
    if ((i+j)%2==parity && Math.random()<0.5){
        //getting the energy
        //let sum=0;
        let s2 = (Math.floor(Math.random() * nspins)+s+1)%nspins
        //this gives a count of all neighbours with spin up
        let up    = (i == 0)?      grid[size-1][j]:grid[i-1][j];
        let down  = (i == size-1)? grid[0][j]     :grid[i+1][j];
        let left  = (j == 0)?      grid[i][size-1]:grid[i][j-1];
        let right = (j == size-1)? grid[i][0]     :grid[i][j+1];
        
        let oldbonds= bonds[s][up]+bonds[s][down]+bonds[s][left]+bonds[s][right]
        
        let newbonds = bonds[s2][up]+bonds[s2][down]+bonds[s2][left]+bonds[s2][right]
        
        //better to have more negative bonds and mew 
        let delta = newbonds-oldbonds+mew[s2]-mew[s];
        //update rule for MCMC
        //I have very little trust in the GPU Math.random function
        if(delta < 0 || 1-Math.random()<Math.exp(-JB*delta)){
        s=s2
        }
    }

    return s;
}, {
        output: [SIZE, SIZE],
        pipeline: true,
        immutable: true
    });

const getval = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
}).setOutput([SIZE, SIZE])




function setpixels(ctx,grid){
    var h = ctx.canvas.height;
    var w = ctx.canvas.width;
    //console.log(h/grid.length)
    scale=h/grid.length
    var imgData = ctx.getImageData(0, 0, w, h);
    var data = imgData.data;  // the array of RGBA values
    //console.log(data.length)
    for(var s = 0; s < data.length; s+=4) {
        x=Math.floor(s/4/w/scale);
        y=Math.floor(((s/4)%w)/scale)
        //s = 4 * x * w + 4 * y    probably
        
        colour= COLOURS[grid[x][y]]
        
        data[s] =     colour[0];
        data[s + 1] = colour[1];
        data[s + 2] = colour[2];
        data[s + 3] = 255;  // fully opaque
    }
    ctx.putImageData(imgData, 0, 0);
}


const $ = q => document.getElementById(q);

const NSPINS=4
const random_new_uniform = gpu.createKernel(function(){
    return -Math.random()
}, {
        output: [NSPINS, NSPINS],
        pipeline: true,
        //immutable: true
    });

const make_symmetric = gpu.createKernel(function(arr){
    let i=this.thread.y
    let j=this.thread.x
    return i>j? arr[i][j]:arr[j][i]
}, {
        output: [NSPINS, NSPINS],
        pipeline: true,
        immutable: true
});

const outputinter = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
}).setOutput([NSPINS, NSPINS])

const bonds = make_symmetric(random_new_uniform())
const COLOURS=[[255,0,0],[0,255,0],[0,0,255],[255,50,255]]
console.log(outputinter(bonds))

var mew=[]
for (var j=0; j<NSPINS;j++){
    mew.push(0);
}


var kT = 2.269
//var mew = 0.0;
var toggle=false;
var kval1=100;
var kval2=100;
var N = 0;
var stepsperframe=1;
var startTime = 0;
var on = false;
$("steps").oninput = function() {
  stepsperframe=Math.pow(4,this.value)*2;
  if (this.value>=0){
  $('stepstext').innerHTML = Math.pow(4,this.value);
  }
  else{
  $('stepstext').innerHTML = "1/"+Math.pow(4,-this.value);
  }
}
$('stopbutton').addEventListener("click", function(){
    on = !on;
    $('stoptext').innerHTML=on? 'Stop':'Start';
    if (on){
        window.requestAnimationFrame(run);
    }
})

$("mew").oninput = function() {
  return
  mew = Math.pow(this.value,3)/25000.0;
  $('mewtext').innerHTML = mew.toFixed(5);
}
    

$("kT").oninput = function() {
  x = Math.pow((this.value/32),3)+0.8194
  kT = Math.exp(x);
  $('kTtext').innerHTML = kT.toFixed(3);
  //console.log(kT);
}


function run(){
    n = Math.random()<=0.5?0:1
    grid2 = Propose(grid,bonds,1/kT,mew,n,NSPINS,SIZE)
    grid = Propose(grid2,bonds,1/kT,mew,1-n,NSPINS,SIZE)
    grid2.delete()
    for (var i=0;i<stepsperframe-1;i++){
        n = Math.random()<=0.5?0:1
        grid2 = Propose(grid,bonds,1/kT,mew,n,NSPINS,SIZE)
        grid.delete()
        grid = Propose(grid2,bonds,1/kT,mew,1-n,NSPINS,SIZE)
        grid2.delete()
    }
    n = Math.random()<=0.5?0:1
    grid2 = Propose(grid,bonds,1/kT,mew,n,NSPINS,SIZE)
    grid.delete()
    grid = getval(grid2);
    grid2.delete()
    //if (Math.random()<0.01){console.log(sum(grid))}
    setpixels(ctx,grid);
    var newtime=(new Date()).getTime()
    var elapsedTime = (newtime - startTime) / 1000;// time in seconds
    startTime=newtime
    $('stepmeter').innerHTML = Number(stepsperframe/elapsedTime).toFixed(0);
    if (on){
        window.requestAnimationFrame(run);
    }
}

INDX=0
grid = zeros([SIZE,SIZE]);
random_ones([SIZE,SIZE],grid,SIZE*SIZE*1/2)
stepsperframe=Math.pow(2,-1)*2;
$('stepstext').innerHTML = "1/"+Math.pow(4,1);
var RGBData;
var NumSpecies=0;
//random_ones([128,128],grid,2000)

var canvas = document.getElementById('grid');
console.log(canvas);
ctx = canvas.getContext("2d");

setpixels(ctx,grid)

window.requestAnimationFrame(run)
