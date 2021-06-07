console.log('hello world');
console.log(Math.random());
console.log([].length)
console.log(Math.pow(2,3));
//first up is probably the rejection free algorithm so I don't have to do any array copying 

//stole this recursive thing but gonna make not recursive
function zeros(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; i++) {
        var arr2= [];
        for (var j=0; j<dimensions[0];j++){
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
    if (grid[x][y]==0){
        grid[x][y]=1
        i++;
    }
 }
}

function sum(arr){
 sx=arr.length;
 sy=arr[0].length;
 s=0
 for (var i=0;i<sx;i++){
     for(var j=0;j<sy;j++){
      s+=arr[i][j]   
     }
 }
 return s
}


function clustering(x,y,grid,xarr,yarr,p,sx,sy,N){
    //it's important to set the grid area to 0 initially so
    //there is no infinite recursion
    if (N>0 && xarr.length>N){return;}
    grid[x][y]=0
    xarr.push(x);
    yarr.push(y);
    //check to move the cluster in each direction
    if (grid[(x+1)%sx][y]==1 && Math.random()<=p){
        clustering((x+1)%sx,y,grid,xarr,yarr,p,sx,sy,N);}
    if (grid[(x-1+sx)%sx][y]==1 && Math.random()<=p){
        clustering((x-1+sx)%sx,y,grid,xarr,yarr,p,sx,sy,N);}
    if (grid[x][(y+1)%sy]==1 && Math.random()<=p){
        clustering(x,(y+1)%sy,grid,xarr,yarr,p,sx,sy,N);}
    if (grid[x][(y-1+sy)%sy]==1 && Math.random()<=p){
        clustering(x,(y-1+sy)%sy,grid,xarr,yarr,p,sx,sy,N);}
}

function make_cluster(x,y,grid,p,N){
    xarr=[];
    yarr=[];
    if (grid[x][y]==1){
        clustering(x,y,grid,xarr,yarr,p,grid.length,grid[0].length,N)
    }
    for (var i=0;i<xarr.length;i++){
        grid[xarr[i]][yarr[i]]=1;
    }
    return [xarr,yarr];
}

function addmod(arr,val,size,pos){
    arr2=[];
    for (var i=0;i<arr.length;i++){
    //please no negative numbers thx
    arr2.push(pos ? (arr[i]+val+size)%size :  (-arr[i]+val+size)%size);
    }
    return arr2
}


function Hamiltonian(grid,u,J){
    sx=grid.length;
    sy=grid[0].length;
    s1=0;
    s2=0;
    for (var x=0;x<sx;x++){
        for (var y=0;y<sy;y++){
            s1+=grid[x][y]*(grid[x][(y+1)%sy]+grid[(x+1)%sx][y]);
            s2+=grid[x][y];
        }
    }
    return u*s2-J*s1
}

function Update(){
    data=Cleaving(grid,1/kT,kval1/100,kval2/100,N,toggle);
    if (data[1].length==0 && data[3].length==0){return;}
    newenergy=Hamiltonian(grid,mew,1)
    delta=newenergy-energy
    A=data[0];
    n = Math.random()
    
    n0 = A*Math.exp(-delta/kT)
    if (n<=n0){
        energy=newenergy;
    }
    else{
        //console.log(n0)
        for (var i=0;i<data[3].length;i++){
            grid[data[3][i]][data[4][i]]=0;
        }
        for (var i=0;i<data[1].length;i++){
            grid[data[1][i]][data[2][i]]=1;
        }
    }

}



function Cleaving(grid,JB,a,b,N,allowdiff){    
    //The equation is: p = max(0,1 − exp[BEc(i,j)−BEI(i,j)])
    //here Ec = -J and EI=0
    //Note this is a ficticious hamiltonian since it considers an overlap
    //to have zero energy whereas in the real one an overlap has infinite
    scaler=a+Math.random()*(b-a)
    // This line clips scaler to be in [0,1]
    scaler= scaler < 0 ? 0: scaler> 1.0 ? 1.0 : scaler
    p=1-Math.exp(-JB*scaler);
    //clip p to be in [0,1)
    p= p < 0 ? 0: p> 0.9999999999999999 ? 0.9999999999999999 :  p

    
    sx=grid.length;
    sy=grid[0].length;
    
    
    if (allowdiff && Math.random()<GRAND_CANONICAL_PROB){
    x=Math.floor(Math.random() * sx);
    y=Math.floor(Math.random() * sy);
    if (grid[x][y]==0){
    grid[x][y]=1;
    //added a particle to the lattice
    return [1,[],[],[x],[y]]
    }
    else{
    grid[x][y]=0;
    //removed a particle from the lattice
    return [1,[x],[y],[],[]]
    }
    }
    
    
    var pos = [];
    for (var i=0;i<sx;i++){
        for (var j=0;j<sy;j++){
            if (grid[i][j]==1){
                pos.push([i,j]);
            }
        }
    }
    if (pos.length==0){return [1,[],[],[],[]];}
    
    i=Math.floor(Math.random() * pos.length);
    x=pos[i][0];
    y=pos[i][1];
    action=Math.floor(Math.random() * 6);
    
    returns=make_cluster(x,y,grid,p,N);
    if (N>0 && xarr.length>N){return [1,[],[],[],[]];}
    xs=returns[0];
    ys=returns[1];
    
    for (var i=0;i<xs.length;i++){
        grid[xs[i]][ys[i]]=0;
    }
    
    //cluster translation
    switch (action){
    case 0:
        rx=addmod(xs,1,sx,true);
        ry=ys;
        break;
    case 1:
        rx=addmod(xs,-1,sx,true);
        ry=ys;
        break;
    case 2:
        rx=xs;
        ry=addmod(ys,1,sy,true);
        break;
    case 3:
        rx=xs;
        ry=addmod(ys,-1,sy,true);
        break;
    case 4:
        //apply rotation to coordinates
        rx=addmod(ys,y+x,sx,false);
        ry=addmod(xs,y-x,sy,true);
        break;
        //rx=(-(ys-y)+x)%sx
        //ry=((xs-x)+y)%sy
    case 5:
        rx=addmod(ys,x-y,sx,true);
        ry=addmod(xs,x+y,sy,false);
        break;
        //rx=((ys-y)+x)%sx
        //ry=(-(xs-x)+y)%sy       
    }
    s=0
    for (var i=0; i<rx.length;i++){
        s+=grid[rx[i]][ry[i]];
    }
    
    if (s!=0){
        for (var i=0; i<xs.length;i++){
            grid[xs[i]][ys[i]]=1
        }
        return [1,[],[],[],[]];
    }
    else{
        qforward=1.0
        qbackward=1.0
        for (var i=0;i<rx.length;i++){
            if (grid[(xs[i]-1+sx)%sx][ys[i]          ]==1){qforward*=(1-p);}
            if (grid[(xs[i]+1)%sx   ][ys[i]          ]==1){qforward*=(1-p);}
            if (grid[xs[i]          ][(ys[i]+1)%sy   ]==1){qforward*=(1-p);}
            if (grid[xs[i]          ][(ys[i]-1+sy)%sy]==1){qforward*=(1-p);}
            if (grid[(rx[i]-1+sx)%sx][ry[i]          ]==1){qbackward*=(1-p);}
            if (grid[(rx[i]+1)%sx   ][ry[i]          ]==1){qbackward*=(1-p);}
            if (grid[rx[i]          ][(ry[i]+1)%sy   ]==1){qbackward*=(1-p);}
            if (grid[rx[i]          ][(ry[i]-1+sy)%sy]==1){qbackward*=(1-p);}
        }
        A=qbackward/qforward;
    
        for (var i=0;i<rx.length;i++){
            grid[rx[i]][ry[i]]=1;
        }   
        
        return [A,xs,ys,rx,ry];
    }
    //console.log(x,y)
    //console.log(xs,ys);
    //console.log(rx,ry);
}

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
        data[s + 1] = 0;//grid[x][y]*255;
        data[s + 2] = grid[x][y]*255;
        data[s + 3] = 255;  // fully opaque
    }
    ctx.putImageData(imgData, 0, 0);
}

const $ = q => document.getElementById(q);
const GRAND_CANONICAL_PROB=0.4
var kT = 1.0
var mew = 0.0;
var toggle=false;
var kval1=100;
var kval2=100;
var N = 0;
var stepsperframe=100;
var startTime = 0;

$("steps").oninput = function() {
  stepsperframe=Math.pow(2,this.value)*grid.length*grid[0].length;
  if (this.value>=0){
  $('stepstext').innerHTML = Math.pow(2,this.value);
  }
  else{
  $('stepstext').innerHTML = "1/"+Math.pow(2,-this.value);
  }
}

$("diff").oninput = function(){
toggle = !toggle;
}

$("mew").oninput = function() {
  mew = this.value/5.0;
  $('mewtext').innerHTML = mew.toFixed(3);
  energy=Hamiltonian(grid,mew,1);
}
    
$("N").oninput = function() {
  N=this.value;
  $('Ntext').innerHTML = N>0?N:'Any';
}
$("kT").oninput = function() {
  kT = Math.exp(this.value/20);
  $('kTtext').innerHTML = kT.toFixed(3);
  size_update();
  //console.log(kT);
}

//both knobs are for the same double slider so they push the other knob out of the way
$("r1").oninput = function() {
    kval1=parseInt(this.value);
    if (kval2<kval1){
    $("r2").value=kval2=kval1;
    }
    size_update();
}
$("r2").oninput = function() {
    kval2=parseInt(this.value);
    if (kval2<kval1){
    $("r1").value=kval1=kval2;
    }
    size_update();
}

function size_update(){
p1=100*kT/kval1//1-Math.exp(-kval1/100/kT)
p2=100*kT/kval2//1-Math.exp(-kval2/100/kT)
$('cstext').innerHTML = p1.toFixed(3)+" to "+p2.toFixed(3);
}


function run(){
 for (var i=0;i<stepsperframe;i++){
 Update(grid,1/kT);
 }
    
 //if (Math.random()<0.01){console.log(sum(grid))}
 setpixels(ctx,grid);
 var newtime=(new Date()).getTime()
 var elapsedTime = (newtime - startTime) / 1000;// time in seconds
 startTime=newtime
 $('stepmeter').innerHTML = Number(stepsperframe/elapsedTime).toFixed(0);
 window.requestAnimationFrame(run);
}

INDX=0
grid = zeros([128,128]);
stepsperframe=Math.pow(2,-7)*grid.length*grid[0].length;
$('stepstext').innerHTML = "1/"+Math.pow(2,7);

random_ones([128,128],grid,2000)
//grid[0][0]=1;
console.log(grid);
console.log((-1+10)%5)
xy=make_cluster(0,0,grid,0,0.8)
console.log(xy);
console.log(grid);
energy=Hamiltonian(grid,1,1)
console.log(energy)

var canvas = document.getElementById('grid');
console.log(canvas);
ctx = canvas.getContext("2d");

setpixels(ctx,grid)

window.requestAnimationFrame(run)
