
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

function ones(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; i++) {
        var arr2= [];
        for (var j=0; j<dimensions[1];j++){
            arr2.push(1);
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

//going to do max test reductions

class Reduction{
    constructor(f){
        this.m2048=  gpu.createKernel(f).setOutput([1024,1024]).setPipeline(true);
        this.m1024 =  gpu.createKernel(f).setOutput([512,512]).setPipeline(true);
        this.m512 =  gpu.createKernel(f).setOutput([256,256]).setPipeline(true);
        this.m256 =  gpu.createKernel(f).setOutput([128,128]).setPipeline(true);
        this.m128 =  gpu.createKernel(f).setOutput([64,64]).setPipeline(true);
        this.m64 =  gpu.createKernel(f).setOutput([32,32]).setPipeline(true);
        this.m32 =  gpu.createKernel(f).setOutput([16,16]).setPipeline(true);
        this.m16 =  gpu.createKernel(f).setOutput([8,8]).setPipeline(true);
        this.m8 =  gpu.createKernel(f).setOutput([4,4]).setPipeline(true);
        this.m4 =  gpu.createKernel(f).setOutput([2,2]).setPipeline(true);
        this.m2=  gpu.createKernel(f,{output: [1, 1],pipeline: true,immutable: true})//.setOutput([1,1]).setPipeline(true);
    }
    call (a){
        return this.m2(this.m4(this.m8(this.m16(this.m32(this.m64(this.m128(this.m256(this.m512(a)))))))))
    }
}


const random_arr = gpu.createKernel(function(offset){
    let i=this.thread.x
    let j=this.thread.y
    if (i>j){
        j=i+1;
        i=-this.thread.y-1
    }
    return j*j+i
}).setOutput([512, 512])



function cpumax(grid){
    var max=-999;
    //var min=999;
    for (var i=0;i<grid.length;i++){
    for (var j=0;j<grid[0].length;j++){
        if (grid[i][j]>max){max=grid[i][j]}
        //if (grid[i][j]<min){min=grid[i][j]}
    }}
    return max//,min
}

function reducemax(a) {
  let i=this.thread.x*2
  let j=this.thread.y*2
  let h1= a[i][j]>a[i][j+1]? a[i][j]:a[i][j+1]
  let h2=a[i+1][j]>a[i+1][j+1]?a[i+1][j]:a[i+1][j+1]
  return h2>h1?h2:h1
}

function reducemin(a) {
  let i=this.thread.x*2
  let j=this.thread.y*2
  let h1= a[i][j]<a[i][j+1]? a[i][j]:a[i][j+1]
  let h2=a[i+1][j]<a[i+1][j+1]?a[i+1][j]:a[i+1][j+1]
  return h2<h1?h2:h1
}

function reduceavg(a) {
  let i=this.thread.x*2
  let j=this.thread.y*2
  return (a[i][j]+a[i][j+1]+a[i+1][j]+a[i+1][j+1])/4.0
}

const maxobj = new Reduction(reducemax)
function gpumax(a){return maxobj.call(a)}

const minobj = new Reduction(reducemin)
function gpumin(a){return minobj.call(a)}

const avgobj = new Reduction(reduceavg)
function gpuavg(a){return avgobj.call(a)}

const ret= gpu.createKernel(function(a){return a[0][0];}).setOutput([1])
const togpularge = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
},{output: [512, 512],pipeline: true,immutable: true})
const squarediff = gpu.createKernel(function(a,avg) {
return (a[this.thread.y][this.thread.x]-avg[0][0])**2;
},{output: [512, 512],pipeline: true})


testarr = random_arr(0.5)

testarrgpu=togpularge(testarr)
var startTime=(new Date()).getTime()

for (var i=0;i<1000;i++){
max = cpumax(testarr)
}

var newtime=(new Date()).getTime()
var ms = (newtime - startTime);//in ms
console.log(max + " on cpu in "+ms+' ms')

max = gpuavg(testarrgpu)

var startTime=(new Date()).getTime()

for (var i=0;i<1000;i++){
max.delete()
max = gpuavg(testarrgpu)
}

var newtime=(new Date()).getTime()
var ms = (newtime - startTime);//in ms
console.log(ret(max) + " on gpu in "+ms+' ms')









const SIZE = 512;
const H = 0.01;
//gpu bound ising proposal using a checkorboard update rule since no neighbours lie on the
//same checkerboard
const Propose = gpu.createKernel(function(grid,JB,mew,parity,size) {
    let i=this.thread.y
    let j=this.thread.x
    let s=grid[i][j];
    //This updates grid cells at least a little bit stochastically, the choice of 0.5 is arbitrary
    //Note: if this is set to 1 you will get weirdness, probably due to the sketchy math.random function
    if ((i+j)%2==parity && Math.random()<0.5){
        //getting the energy
        let sum=0;
        //this gives a count of all neighbours with spin up
        sum+= (i == 0)?      grid[size-1][j]:grid[i-1][j];
        sum+= (i == size-1)? grid[0][j]     :grid[i+1][j];
        sum+= (j == 0)?      grid[i][size-1]:grid[i][j-1];
        sum+= (j == size-1)? grid[i][0]     :grid[i][j+1];
        let delta = 2.0 * (grid[i][j]*2-1)*(2*sum-4+mew) ;
        //update rule for MCMC
        //I have very little trust in the GPU Math.random function
        if(delta < 0 || 1-Math.random()<Math.exp(-JB*delta)){
        s=1-grid[i][j];
        }
    }

    return s;
}, {
        output: [SIZE, SIZE],
        pipeline: true,
        immutable: true
    });

const Ey = gpu.createKernel(function(V,mutable,h,size) {
    let i=this.thread.y
    let j=this.thread.x
    let ip1= (i == size-1)? 0     :i+1;
    let jp1= (j == size-1)? 0     :j+1;
    if (mutable[i][j]+mutable[ip1][j]+mutable[i][jp1]+mutable[ip1][jp1]>0){
        return -(V[ip1][jp1]-V[ip1][j]+V[i][jp1]-V[i][j])/(2*h);
    }
    return 0;
    
},{output: [SIZE, SIZE],pipeline: true})

const Ex = gpu.createKernel(function(V,mutable,h,size) {
    let i=this.thread.y
    let j=this.thread.x
    let ip1= (i == size-1)? 0     :i+1;
    let jp1= (j == size-1)? 0     :j+1;
    if (mutable[i][j]+mutable[ip1][j]+mutable[i][jp1]+mutable[ip1][jp1]>0){
        return -(V[ip1][jp1]-V[i][jp1]+V[ip1][j]-V[i][j])/(2*h);
    }
    return 0;
    
},{output: [SIZE, SIZE],pipeline: true})

const M = gpu.createKernel(function(Ex,Ey) {
    let i=this.thread.y
    let j=this.thread.x
    return Math.sqrt(Ex[i][j]*Ex[i][j]+Ey[i][j]*Ey[i][j]);
    
},{output: [SIZE, SIZE],pipeline: true})


const getval = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
}).setOutput([SIZE, SIZE])

const togpu = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
},{output: [SIZE, SIZE],pipeline: true,immutable: true})

const toperm = gpu.createKernel(function(a) {
return 6-4*a[this.thread.y][this.thread.x];
},{output: [SIZE, SIZE],pipeline: true})

const SOR = gpu.createKernel(function(V,p,eps,mutable,w,h,parity,size) {
    let i=this.thread.y
    let j=this.thread.x
    let s=V[i][j];
    if ((i+j)%2==parity && mutable[i][j]==1){
        //getting the energy
        
        let im1= (i == 0)?      size-1:i-1;
        let ip1= (i == size-1)? 0     :i+1;
        let jm1= (j == 0)?      size-1:j-1;
        let jp1= (j == size-1)? 0     :j+1;
        let a0=eps[i][j]+eps[im1][j]+eps[i][jm1]+eps[im1][jm1]
        let a1=(eps[i][j]+eps[i][jm1])/2.0
        let a2=(eps[im1][j]+eps[i][j])/2.0
        let a3=(eps[im1][jm1]+eps[im1][j])/2.0
        let a4=(eps[i][jm1]+eps[im1][jm1])/2.0
        let R = (h**2*p[i][j]+a1*V[ip1][j]+a2*V[i][jp1]+a3*V[im1][j]+a4*V[i][jm1])/a0-V[i][j]

        s=V[i][j]+w*R
    }

    return s;
}, {
        output: [SIZE, SIZE],
        pipeline: true,
        immutable: true,
    });





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
        data[s] = grid[x][y]*255;
        data[s + 1] = grid[x][y]*255;//0;
        data[s + 2] = grid[x][y]*255;
        data[s + 3] = 255;  // fully opaque
    }
    ctx.putImageData(imgData, 0, 0);
}

function Vmap(ctx,grid){
    var h = ctx.canvas.height;
    var w = ctx.canvas.width;
    //console.log(h/grid.length)
    scale=h/grid.length
    var imgData = ctx.getImageData(0, 0, w, h);
    var data = imgData.data;  // the array of RGBA values
    //console.log(data.length)
    //console.log(data[0])
    var max=-999;
    var min=999;
    for (var i=0;i<grid.length;i++){
    for (var j=0;j<grid[0].length;j++){
        if (grid[i][j]>max){max=grid[i][j]}
        if (grid[i][j]<min){min=grid[i][j]}
    }}
    for(var s = 0; s < data.length; s+=4) {
        x=Math.floor(s/4/w/scale);
        y=Math.floor(((s/4)%w)/scale)
        //s = 4 * x * w + 4 * y    probably
        var gij=(grid[x][y]-min)/(max-min)
        let R = gij<0.35?0:gij<0.66? (gij-0.35)/0.31:gij<0.89?1:1-0.5*(gij-0.89)/0.11
        let G = gij<0.125?0:gij<0.375?(gij-0.125)/0.25:gij<0.64?1:gij<0.91?1-(gij-0.64)/0.27:0
        let B = gij<0.11?0.5+gij/0.22:gij<0.34?1:gij<0.65?1-(gij-0.34)/0.31:0
        data[s] =     R*255;
        data[s + 1] = G*255;
        data[s + 2] = B*255;
        data[s + 3] = 255;  // fully opaque
    }
    //console.log(data[0]);
    ctx.putImageData(imgData, 0, 0);
    //console.log('hwat?')
}

const to2D= gpu.createKernel(function(a){return a;},{output: [1,1],pipeline: true,immutable: true})
const eqblock= gpu.createKernel(function(a,R){
    let i=this.thread.x
    //coordinate system is upside down and sideways jesus
    let j=this.thread.y
    let ri=Math.floor(i/R/2)*2*R+R
    let rj=Math.floor(j/R/2)*2*R+R
    return a[ri][rj]
},{output: [512,512],pipeline: true})

const FAKEMAX=to2D(1.8)
function Erender(ctx,V){
    EX=Ex(V,mutable,H,SIZE)
    EY=Ey(V,mutable,H,SIZE)
    mag=M(EX,EY)
    amax=gpumax(eqblock(mag,16))
    max=gpumax(mag)
    min=gpumin(mag)
    avg=gpuavg(mag)
    variance=gpuavg(squarediff(mag,avg))
    Emap(EX,EY,mag,max,min,avg,variance,amax)
    ctx.drawImage(Emap.getCanvas(),0,0)
    //console.log(ret(avg)[0],ret(variance)[0]**0.5)
    max.delete()
    min.delete()
    avg.delete()
    amax.delete()
    variance.delete()
}

const Emap = gpu.createKernel(function(Ex,Ey,M,maxarr,minarr,avg,variance,amax) {
    let j=this.thread.x
    //coordinate system is upside down and sideways jesus
    let i=511-this.thread.y
    let max=maxarr[0][0];
    let min=minarr[0][0];
    let A=1.0;
    let R=16
    let ri=Math.floor(i/R/2)*2*R+R
    let rj=Math.floor(j/R/2)*2*R+R
    //coordintates of potential vector arrow this is part of
    
    //for(let i1=0;i1<9;i1++){
    //ri=(Math.floor(i/R/2+i1/3-1)*2*R+R)%512
    //rj=(Math.floor(j/R/2+i1%3-1)*2*R+R)%512
        
    //vector representing the field arrow
    let dx=-(R-7)*Ex[ri][rj]/amax[0][0]
    let dy=(R-7)*Ey[ri][rj]/amax[0][0]
    let m=(R-7)*(M[ri][rj])/amax[0][0]
    let hs=m>3.5?3.5:m
    //now we check if our pixel is inside the arrow
    if (m>0){
        let x = i-ri
        let y=j-rj
        //arrowhead can be done using inf and 1 norm in rotated coords
        let X = dx*(dy-y)/m - dy*(dx-x)/m
        let Y = dy*(dy-y)/m + dx*(dx-x)/m
        
        // A<=0 if its in the arrowhead, 0<A<1 if its on the border A>1 if outside.
        A=Math.min(A,Math.max(-(hs-1.5*Math.abs(X)-0.5*Math.abs(Y)), -(hs-Math.abs(Y+hs))));
        
        //arrow body done by looking at length of orthogonal (to E field) part of pixel vector
        let d=x*dx+y*dy
        X = x-d*dx/m/m
        Y = y-d*dy/m/m       
        if (d<m*m){
            //Same assignment here but with the arrow body
            A=Math.min(A,Math.max(-3*d/hs,1-(hs/3.2-(X*X+Y*Y)**0.5)/0.4))
        }
        //clip A between 0 and 1
        A=A<0?0:A
    }
    //}
    //use mean and variance here
    let stdv=variance[0][0]**0.5
    max=max>avg[0][0]+3*stdv?avg[0][0]+3*stdv:max
    min=min<avg[0][0]-3*stdv?avg[0][0]-3*stdv:min
    //let gij=(M[i][j]-avg[0][0]+3*stdv)/6/stdv
    let gij=(M[i][j]-min)/(max-min)
    gij=gij<0?0:gij>1?1.1-0.272*Math.exp(-gij):gij
    //already used R for something woops
    R = gij<0.35?0:gij<0.66? (gij-0.35)/0.31:gij<0.89?1:1-0.5*(gij-0.89)/0.11
    let G = gij<0.125?0:gij<0.375?(gij-0.125)/0.25:gij<0.64?1:gij<0.91?1-(gij-0.64)/0.27:0
    let B = gij<0.11?0.5+gij/0.22:gij<0.34?1:gij<0.65?1-(gij-0.34)/0.31:0
    this.color(R*A,G*A,B*A, 1);
    
}).setOutput([512, 512]).setGraphical(true);

/*
Maybe I write general cmap code later
_jet_data = {'red':   ((0.00, 0, 0),
                       (0.35, 0, 0),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.5, 0.5)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0, 0),
                       (1.000, 0, 0)),
             'blue':  ((0.00, 0.5, 0.5),
                       (0.11, 1, 1),
                       (0.34, 1, 1),
                       (0.65, 0, 0),
                       (1.00, 0, 0))}


*/

function smallcap(){
    V.delete()
    mutable.delete()
    V = zeros([SIZE,SIZE]);
    mutable=ones([SIZE,SIZE]);
    for (var j=184;j<SIZE-184;j++){
        mutable[184][j]=0;
        mutable[SIZE-185][j]=0;
        V[184][j]=1;
        V[SIZE-185][j]=-1;
    }
    mutable=togpu(mutable)
    V = togpu(V);
}
function bigcap(){
    V.delete()
    mutable.delete()
    V = zeros([SIZE,SIZE]);
    mutable=ones([SIZE,SIZE]);
    for (var j=0;j<SIZE;j++){
        mutable[0][j]=0;
        mutable[SIZE-1][j]=0;
        V[0][j]=1;
        V[SIZE-1][j]=-1;
    }
    mutable=togpu(mutable)
    V = togpu(V);
}



const $ = q => document.getElementById(q);
var kT = 2.269
var mew = 0.0;
var toggle=false;
var toggle2=false;
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
  mew = Math.pow(this.value,3)/128000.0;
  $('mewtext').innerHTML = mew.toFixed(5);
}
    

$("kT").oninput = function() {
  x = Math.pow((this.value/32),3)+0.8194
  kT = Math.exp(x);
  $('kTtext').innerHTML = kT.toFixed(3);
  //console.log(kT);
}
$("PlotV").oninput = function(){
toggle = !toggle;
}
$("SmallC").oninput = function(){
toggle2 = !toggle2;
    if (toggle2){smallcap()} else{bigcap()}
}

function nextVoltage(steps,eps){
for (var i=0;i<steps;i++){
    V2= SOR(V,p,eps,mutable,w,H,0,SIZE)
    V.delete()
    V = SOR(V2,p,eps,mutable,w,H,1,SIZE)
    V2.delete()
}
}


function run(){
    //TODO verify there isn't a memory leak with grid and grid2.
    n = Math.random()<=0.5?0:1
    grid = Propose(grid2,1/kT,mew,1-n,SIZE)
    grid2.delete()
    for (var i=0;i<stepsperframe-1;i++){
        n = Math.random()<=0.5?0:1
        grid2 = Propose(grid,1/kT,mew,n,SIZE)
        grid.delete()
        grid = Propose(grid2,1/kT,mew,1-n,SIZE)
        grid2.delete()
    }
    n = Math.random()<=0.5?0:1
    grid2 = Propose(grid,1/kT,mew,n,SIZE)
    grid.delete()
    
    toperm(grid);
    nextVoltage(400,eps);
    if (!toggle){
    Erender(ctx2,V);
    }
    else{Vmap(ctx2,getval(V));}
    
    // getval copies grid2 to the cpu
    setpixels(ctx,getval(grid2));
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
var canvas2 = document.getElementById('Volts');
ctx2 = canvas2.getContext("2d");


setpixels(ctx,grid)
//start the pipeline
grid2 = Propose(grid,1/kT,mew,0,SIZE)
V = zeros([SIZE,SIZE]);
mutable=ones([SIZE,SIZE]);


for (var j=0;j<SIZE;j++){
    mutable[0][j]=0;
    mutable[SIZE-1][j]=0;
    V[0][j]=1;
    V[SIZE-1][j]=-1;
}



mutable=togpu(mutable)
V = togpu(V);
p=togpu(zeros([SIZE,SIZE]));
t = 2*Math.cos(Math.PI/SIZE)
w = (8-Math.sqrt(64-16*t*t))/t/t
console.log(w)
eps=toperm(grid);

console.log(getval(V))
console.log(getval(p))
console.log(getval(mutable))
console.log(Math.max(1,2,3,67))
console.log(2**2);
nextVoltage(200,eps);
console.log(getval(V));
Vmap(ctx2,getval(V));
window.requestAnimationFrame(run)
