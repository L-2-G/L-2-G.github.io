
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

function Update(){
    data=Proposal(grid,2,0,mew);
    x=data[0];
    y=data[1];
    val=data[2]
    A=data[3];
    delta=data[4];
    
    n = Math.random()
    
    n0 = A*Math.exp(-delta/kT)
    if (n<=n0){
        grid[x][y]=val;
    }

}
function Proposal(grid,Jd,Ju,u){
    sx=grid.length;
    sy=grid[0].length;
    //first sy are already filled in and ommited from deposition
    Q=NumSpecies-sy*PREFILLED
    x=PREFILLED+Math.floor(Math.random() * (sx-PREFILLED));
    y=Math.floor(Math.random() * sy);
    var delta;
    var A;
    var particle;
    if (grid[x][y]==0){
    particle=sy*PREFILLED+1+Math.floor(Math.random() * Q);
    bonds=bondtypes(grid,x,y,sx,sy,particle);
    delta=u-Jd*bonds[0]-Ju*bonds[1]
    A=Q;
    }
    else{
    bonds=bondtypes(grid,x,y,sx,sy,grid[x][y]);
    delta=-u+Jd*bonds[0]+Ju*bonds[1]
    A=1/Q
    particle = 0;
    }
    return [x,y,particle,A,delta]
}

function bondtypes(grid,i,j,sx,sy,val){
    good=0;
    bad=0;
    //get ready for a ton of if statements because this will be done by each case
    //value format is i*w+j+1
    //w=sy
    //no periodic boundaries for vertical direction which is actually x
    if(i>0 && grid[i-1][j]!=0){
        //given i*w+j+1 find (i-1)*w+j+1 ===  i*w+j+1 - w
        goodval=val-sy
        if (goodval==grid[i-1][j]){
            good++;
        }
        else{bad++;}
    }
    if(i<sx-1 && grid[i+1][j]!=0){
        //given i*w+j+1 find (i-1)*w+j+1 ===  i*w+j+1 - w
        goodval=val+sy
        if (goodval==grid[i+1][j]){
            good++;
        }
        else{bad++;}
    }
    jplus=j+1<sy?j+1:0
    jminus=j-1>=0?j-1:sy-1
    if(grid[i][jminus]!=0){
        //given i*w+j+1 find i*w+(j-1)%sy+1
        jp1=val%sy;
        goodval=val-1;
        if (jp1==1){goodval+=sy}
        if (goodval==grid[i][jminus]){
            good++;
        }
        else{bad++;}
    }
    if(grid[i][jplus]!=0){
        //given i*w+j+1 find i*w+(j+1)%sy+1
        jp1=val%sy;
        goodval=val+1;
        if (jp1==0){goodval-=sy}
        if (goodval==grid[i][jplus]){
            good++;
        }
        else{bad++;}
    }
    return [good,bad];
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
        if (grid[x][y]>0){
        s2=(grid[x][y]-1)*4;
        data[s] = RGBData[s2];
        data[s + 1] = RGBData[s2+1];
        data[s + 2] = RGBData[s2+2];
        data[s + 3] = RGBData[s2+3];  // fully opaque
        }
        else{
        data[s] = 0
        data[s + 1] = 0
        data[s + 2] = 0
        data[s + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

const $ = q => document.getElementById(q);
const GRAND_CANONICAL_PROB=0.4
const PREFILLED=1;
const MAXDIMX=48;
var kT = 1.0
var mew = 0.0;
var toggle=false;
var kval1=100;
var kval2=100;
var N = 0;
var stepsperframe=100;
var startTime = 0;
var on = false;
$("steps").oninput = function() {
  stepsperframe=Math.pow(2,this.value)*grid.length*grid[0].length;
  if (this.value>=0){
  $('stepstext').innerHTML = Math.pow(2,this.value);
  }
  else{
  $('stepstext').innerHTML = "1/"+Math.pow(2,-this.value);
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
  mew = this.value/5.0;
  $('mewtext').innerHTML = mew.toFixed(3);
}
    

$("kT").oninput = function() {
  kT = Math.exp(this.value/20);
  $('kTtext').innerHTML = kT.toFixed(3);
  //console.log(kT);
}
document.onpaste = function(pasteEvent) {
    // consider the first item (can be easily extended for multiple items)
    var item = pasteEvent.clipboardData.items[0];
    var img = new Image();
    if (item.type.indexOf("image") === 0)
    {
        var blob = item.getAsFile();
 
        var reader = new FileReader();
        reader.onload = function(event) {
            //img = $("imcontainer");
            img.src = event.target.result;
            img.onload = function(){
                w=img.width;
                h=img.height
                if (w>MAXDIMX){   
                    h = Math.floor(h*MAXDIMX/w);
                    w=MAXDIMX;
                }
                scale=Math.floor(512/w);
                canvas.width=w*scale;
                canvas.height=h*scale;
                ctx.drawImage(img, 0, 0,w,h);
                grid = zeros([h,w]);
                var imgData = ctx.getImageData(0, 0, w, h);
                RGBData = imgData.data; 
                NumSpecies=h*w;
                console.log("sx:"+grid.length +" sy:"+grid[0].length)
                //fill in some starting positions
                //by setting bounds for i and j you specify the particles already in the right place
                for (var i=0;i<PREFILLED;i++){
                    for (var j=0;j<w;j++){
                        grid[i][j]=i*w+j+1;
                    }
                }
                setpixels(ctx,grid)
            }

        };
 
        reader.readAsDataURL(blob);
    }
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
 if (on){
     window.requestAnimationFrame(run);
 }
}

INDX=0
grid = zeros([128,128]);
stepsperframe=Math.pow(2,-1)*grid.length*grid[0].length;
$('stepstext').innerHTML = "1/"+Math.pow(2,1);
var RGBData;
var NumSpecies=0;
//random_ones([128,128],grid,2000)

var canvas = document.getElementById('grid');
console.log(canvas);
ctx = canvas.getContext("2d");

setpixels(ctx,grid)

window.requestAnimationFrame(run)
