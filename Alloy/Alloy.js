
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
        arr[x][y]=Math.floor(Math.random()*3)+1
        i++;
    }
 }
}

const gpu = new GPU();

const SIZE = 256;


//gpu bound ising proposal using a checkorboard update rule since no neighbours lie on the
//same checkerboard
const choose_perm = gpu.createKernel(function(grid,bonds,JB,perm,ip,jp,nspins,s) {
    let i=this.thread.y*4+ip
    let j=this.thread.x*4+jp

    let returnval=0;
    //choose the permutation
    let mutate = (Math.floor(Math.random() * 23))+1
    //evaluate current bonds
    
    let a21=grid[(i+1)%s][j%s]
    let a31=grid[(i+2)%s][j%s]
    
    let a42=grid[(i+3)%s][(j+1)%s]
    let a43=grid[(i+3)%s][(j+2)%s]
    
    let a12=grid[i%s][(j+1)%s]
    let a13=grid[i%s][(j+2)%s]
    
    let a24=grid[(i+1)%s][(j+3)%s]
    let a34=grid[(i+2)%s][(j+3)%s]
     
    
    let a22=grid[(i+1)%s][(j+1)%s]
    let a23=grid[(i+1)%s][(j+2)%s]
    let a32=grid[(i+2)%s][(j+1)%s]
    let a33=grid[(i+2)%s][(j+2)%s]
     
    
    //left and bottom
    let oldbonds = bonds[a21][a22]+bonds[a31][a32]+bonds[a42][a32]+bonds[a43][a33]
    //right and top
    oldbonds+= bonds[a22][a12]+bonds[a23][a13]+bonds[a23][a24]+bonds[a33][a34]
    //inner
    oldbonds+= bonds[a22][a23]+bonds[a22][a32]+bonds[a33][a32]+bonds[a33][a23]
    
    //now we schmix up a22,a23,a32,a33
    let schmixer=[a22,a23,a32,a33]
    a22=schmixer[perm[mutate][0]]
    a23=schmixer[perm[mutate][1]]
    a32=schmixer[perm[mutate][2]]
    a33=schmixer[perm[mutate][3]]
    
    //re-evaluate all the bonds
    let newbonds = bonds[a21][a22]+bonds[a31][a32]+bonds[a42][a32]+bonds[a43][a33]
    //right and top
    newbonds+= bonds[a22][a12]+bonds[a23][a13]+bonds[a23][a24]+bonds[a33][a34]
    //inner
    newbonds+= bonds[a22][a23]+bonds[a22][a32]+bonds[a33][a32]+bonds[a33][a23]
    
    let delta = newbonds-oldbonds//change this
    //update rule for MCMC
    //I have very little trust in the GPU Math.random function
    if(delta < 0 || 1-Math.random()<Math.exp(-JB*delta)){
    returnval=mutate
    }

    return returnval;
}, {
        output: [SIZE/4, SIZE/4],
        pipeline: true,
        //immutable: true
    });

const finalize = gpu.createKernel(function(grid,updates,inverseperm,ip,jp,size) {
    //my i and j index
    let i=this.thread.y
    let j=this.thread.x
    
    if ((i-ip+4)%4==0||(i-ip+4)%4==3||(j-jp+4)%4==0||(j-jp+4)%4==3){
    return grid[i][j];
    }
    
    let sp=Math.floor(size/4)
    //the grid containing permutations i and j spot
    let y = Math.floor((i-ip)/4+sp)%sp
    let x = Math.floor((j-jp)/4+sp)%sp
    
    //the top left of this grid section
    let io=y*4+ip
    let jo=x*4+jp
    //my relative position to the top left of the movable parts
    let myioff = (i-ip+4)%4-1
    let myjoff = (j-jp+4)%4-1
    //my index in the original permutation
    let mynum=2*myioff+myjoff
    
    /*
    let a22=grid[(i+1)%s][(j+1)%s] -> i0 j0 pos
    let a23=grid[(i+1)%s][(j+2)%s] -> i0 j1 pos
    let a32=grid[(i+2)%s][(j+1)%s] -> i1 j0 pos
    let a33=grid[(i+2)%s][(j+2)%s] -> i1 j1 pos
    */
    
    //need to figure out which one was moved to this position and return that spot
    let invperm = inverseperm[updates[y][x]][mynum]
    //the target's relative position to the top left movable bit
    let ioff=invperm>>1
    let joff=invperm%2
    // the target's position
    let reti=(i+ioff-myioff+size)%size
    let retj=(j+joff-myjoff+size)%size
    //this is the one that got moved into grid[i][j] in chooseperm
    return grid[reti][retj];
}, {
        output: [SIZE, SIZE],
        pipeline: true,
        immutable: true
    });

console.log(0>>1,0%2)//0 0
console.log(1>>1,1%2)//0 1
console.log(2>>1,2%2)//1 0
console.log(3>>1,3%2)//1 1


//gpu bound ising proposal using a checkorboard update rule since no neighbours lie on the
//same checkerboard
const ProposeOLD = gpu.createKernel(function(grid,bonds,JB,mew,parity,nspins,size) {
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

permutations = []
inversepermutations = []
for (var i=0; i<4; i++){
for (var j=(i+1)%4;j!=i;j=(j+1)%4){
for (var k=(i+1)%4;k!=i;k=(k+1)%4){
for (var l=(i+1)%4;l!=i;l=(l+1)%4){    
    if (j!=k && k!=l && l!=j){
        permutations.push([i,j,k,l])
        inversepermutations.push([0,0,0,0])
    }
}
}
}
}


for (var idx=0;idx<24;idx++){
for (var i=0;i<4;i++){
inversepermutations[idx][permutations[idx][i]]=i
}
}
console.log(permutations)
console.log(inversepermutations)

const toconstperm = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
    },{ output: [24, 4],
        pipeline: true,
        immutable: true})

const PERM = permutations//toconstperm(permutations)
const INVPERM= inversepermutations//toconstperm(inversepermutations)

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

const TESTARRCONST = gpu.createKernel(function() {
let b = [0,1]
return b[this.thread.y%2]
}).setOutput([4, 4])

console.log(TESTARRCONST())

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
  kT = Math.exp(x)*0.05;
  $('kTtext').innerHTML = kT.toFixed(3);
  //console.log(kT);
}


function run(){
    ip=Math.floor(Math.random()*4)
    jp=Math.floor(Math.random()*4)
    choose_perm(grid,bonds,1/kT,PERM,ip,jp,NSPINS,SIZE)
    grid2 = finalize(grid,choices,INVPERM,ip,jp,SIZE) 
    
    
    //setpixels(ctx,getval(grid2));
    //return
    
    
    ip=Math.floor(Math.random()*4)
    jp=Math.floor(Math.random()*4)
    choose_perm(grid2,bonds,1/kT,PERM,ip,jp,NSPINS,SIZE)
    grid = finalize(grid2,choices,INVPERM,ip,jp,SIZE) 
    //n = Math.random()<=0.5?0:1
    //grid2 = Propose(grid,bonds,1/kT,mew,n,NSPINS,SIZE)
    //grid = Propose(grid2,bonds,1/kT,mew,1-n,NSPINS,SIZE)
    grid2.delete()
    for (var i=0;i<stepsperframe-1;i++){
        
        ip=Math.floor(Math.random()*4)
        jp=Math.floor(Math.random()*4)
        choose_perm(grid,bonds,1/kT,PERM,ip,jp,NSPINS,SIZE)
        grid2 = finalize(grid,choices,INVPERM,ip,jp,SIZE) 
        grid.delete()
        
        ip=Math.floor(Math.random()*4)
        jp=Math.floor(Math.random()*4)
        choose_perm(grid2,bonds,1/kT,PERM,ip,jp,NSPINS,SIZE)
        grid = finalize(grid2,choices,INVPERM,ip,jp,SIZE) 
        grid2.delete()
    }
    ip=Math.floor(Math.random()*4)
    jp=Math.floor(Math.random()*4)
    choose_perm(grid,bonds,1/kT,PERM,ip,jp,NSPINS,SIZE)
    grid2 = finalize(grid,choices,INVPERM,ip,jp,SIZE) 
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

choices = choose_perm(grid,bonds,1/kT,PERM,0,0,NSPINS,SIZE)
const getval2 = gpu.createKernel(function(a) {
return a[this.thread.y][this.thread.x];
}).setOutput([SIZE/4, SIZE/4])

console.log(getval2(choices))

var canvas = document.getElementById('grid');
console.log(canvas);
ctx = canvas.getContext("2d");

setpixels(ctx,grid)

//window.requestAnimationFrame(run)
