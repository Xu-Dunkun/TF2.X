??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??	
?
mnist_model/conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namemnist_model/conv1/kernel
?
,mnist_model/conv1/kernel/Read/ReadVariableOpReadVariableOpmnist_model/conv1/kernel*&
_output_shapes
:*
dtype0
?
mnist_model/conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namemnist_model/conv1/bias
}
*mnist_model/conv1/bias/Read/ReadVariableOpReadVariableOpmnist_model/conv1/bias*
_output_shapes
:*
dtype0
?
mnist_model/bn1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namemnist_model/bn1/gamma
{
)mnist_model/bn1/gamma/Read/ReadVariableOpReadVariableOpmnist_model/bn1/gamma*
_output_shapes
:*
dtype0
?
mnist_model/bn1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namemnist_model/bn1/beta
y
(mnist_model/bn1/beta/Read/ReadVariableOpReadVariableOpmnist_model/bn1/beta*
_output_shapes
:*
dtype0
?
mnist_model/bn1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namemnist_model/bn1/moving_mean
?
/mnist_model/bn1/moving_mean/Read/ReadVariableOpReadVariableOpmnist_model/bn1/moving_mean*
_output_shapes
:*
dtype0
?
mnist_model/bn1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!mnist_model/bn1/moving_variance
?
3mnist_model/bn1/moving_variance/Read/ReadVariableOpReadVariableOpmnist_model/bn1/moving_variance*
_output_shapes
:*
dtype0
?
mnist_model/f1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*&
shared_namemnist_model/f1/kernel
?
)mnist_model/f1/kernel/Read/ReadVariableOpReadVariableOpmnist_model/f1/kernel* 
_output_shapes
:
?	?*
dtype0

mnist_model/f1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namemnist_model/f1/bias
x
'mnist_model/f1/bias/Read/ReadVariableOpReadVariableOpmnist_model/f1/bias*
_output_shapes	
:?*
dtype0
?
mnist_model/f2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_namemnist_model/f2/kernel
?
)mnist_model/f2/kernel/Read/ReadVariableOpReadVariableOpmnist_model/f2/kernel*
_output_shapes
:	?
*
dtype0
~
mnist_model/f2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namemnist_model/f2/bias
w
'mnist_model/f2/bias/Read/ReadVariableOpReadVariableOpmnist_model/f2/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/mnist_model/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/mnist_model/conv1/kernel/m
?
3Adam/mnist_model/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/conv1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/mnist_model/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/mnist_model/conv1/bias/m
?
1Adam/mnist_model/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/conv1/bias/m*
_output_shapes
:*
dtype0
?
Adam/mnist_model/bn1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mnist_model/bn1/gamma/m
?
0Adam/mnist_model/bn1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/bn1/gamma/m*
_output_shapes
:*
dtype0
?
Adam/mnist_model/bn1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/mnist_model/bn1/beta/m
?
/Adam/mnist_model/bn1/beta/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/bn1/beta/m*
_output_shapes
:*
dtype0
?
Adam/mnist_model/f1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*-
shared_nameAdam/mnist_model/f1/kernel/m
?
0Adam/mnist_model/f1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f1/kernel/m* 
_output_shapes
:
?	?*
dtype0
?
Adam/mnist_model/f1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/mnist_model/f1/bias/m
?
.Adam/mnist_model/f1/bias/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/mnist_model/f2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*-
shared_nameAdam/mnist_model/f2/kernel/m
?
0Adam/mnist_model/f2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f2/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/mnist_model/f2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/mnist_model/f2/bias/m
?
.Adam/mnist_model/f2/bias/m/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f2/bias/m*
_output_shapes
:
*
dtype0
?
Adam/mnist_model/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/mnist_model/conv1/kernel/v
?
3Adam/mnist_model/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/conv1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/mnist_model/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/mnist_model/conv1/bias/v
?
1Adam/mnist_model/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/conv1/bias/v*
_output_shapes
:*
dtype0
?
Adam/mnist_model/bn1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/mnist_model/bn1/gamma/v
?
0Adam/mnist_model/bn1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/bn1/gamma/v*
_output_shapes
:*
dtype0
?
Adam/mnist_model/bn1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/mnist_model/bn1/beta/v
?
/Adam/mnist_model/bn1/beta/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/bn1/beta/v*
_output_shapes
:*
dtype0
?
Adam/mnist_model/f1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*-
shared_nameAdam/mnist_model/f1/kernel/v
?
0Adam/mnist_model/f1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f1/kernel/v* 
_output_shapes
:
?	?*
dtype0
?
Adam/mnist_model/f1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameAdam/mnist_model/f1/bias/v
?
.Adam/mnist_model/f1/bias/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/mnist_model/f2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*-
shared_nameAdam/mnist_model/f2/kernel/v
?
0Adam/mnist_model/f2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f2/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/mnist_model/f2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/mnist_model/f2/bias/v
?
.Adam/mnist_model/f2/bias/v/Read/ReadVariableOpReadVariableOpAdam/mnist_model/f2/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
c1
b1
a1
p1
d1
flatten
f1
d2
	f2

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratem?m?m?m?/m?0m?9m?:m?v?v?v?v?/v?0v?9v?:v?
 
F
0
1
2
3
4
5
/6
07
98
:9
8
0
1
2
3
/4
05
96
:7
?

Dlayers
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
	variables
Gmetrics
trainable_variables
Hlayer_metrics
 
RP
VARIABLE_VALUEmnist_model/conv1/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEmnist_model/conv1/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Ilayers
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
	variables
Lmetrics
trainable_variables
Mlayer_metrics
 
NL
VARIABLE_VALUEmnist_model/bn1/gamma#b1/gamma/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEmnist_model/bn1/beta"b1/beta/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmnist_model/bn1/moving_mean)b1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEmnist_model/bn1/moving_variance-b1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
?

Nlayers
regularization_losses
Onon_trainable_variables
Player_regularization_losses
	variables
Qmetrics
trainable_variables
Rlayer_metrics
 
 
 
?

Slayers
regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
 	variables
Vmetrics
!trainable_variables
Wlayer_metrics
 
 
 
?

Xlayers
#regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
$	variables
[metrics
%trainable_variables
\layer_metrics
 
 
 
?

]layers
'regularization_losses
^non_trainable_variables
_layer_regularization_losses
(	variables
`metrics
)trainable_variables
alayer_metrics
 
 
 
?

blayers
+regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
,	variables
emetrics
-trainable_variables
flayer_metrics
OM
VARIABLE_VALUEmnist_model/f1/kernel$f1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmnist_model/f1/bias"f1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?

glayers
1regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
2	variables
jmetrics
3trainable_variables
klayer_metrics
 
 
 
?

llayers
5regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
6	variables
ometrics
7trainable_variables
player_metrics
OM
VARIABLE_VALUEmnist_model/f2/kernel$f2/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmnist_model/f2/bias"f2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
?

qlayers
;regularization_losses
rnon_trainable_variables
slayer_regularization_losses
<	variables
tmetrics
=trainable_variables
ulayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8

0
1
 

v0
w1
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	xtotal
	ycount
z	variables
{	keras_api
E
	|total
	}count
~
_fn_kwargs
	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

z	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

	variables
us
VARIABLE_VALUEAdam/mnist_model/conv1/kernel/m@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/mnist_model/conv1/bias/m>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/mnist_model/bn1/gamma/m?b1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/mnist_model/bn1/beta/m>b1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/mnist_model/f1/kernel/m@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/mnist_model/f1/bias/m>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/mnist_model/f2/kernel/m@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/mnist_model/f2/bias/m>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/mnist_model/conv1/kernel/v@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/mnist_model/conv1/bias/v>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/mnist_model/bn1/gamma/v?b1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/mnist_model/bn1/beta/v>b1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/mnist_model/f1/kernel/v@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/mnist_model/f1/bias/v>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/mnist_model/f2/kernel/v@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/mnist_model/f2/bias/v>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1mnist_model/conv1/kernelmnist_model/conv1/biasmnist_model/bn1/gammamnist_model/bn1/betamnist_model/bn1/moving_meanmnist_model/bn1/moving_variancemnist_model/f1/kernelmnist_model/f1/biasmnist_model/f2/kernelmnist_model/f2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_23804
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,mnist_model/conv1/kernel/Read/ReadVariableOp*mnist_model/conv1/bias/Read/ReadVariableOp)mnist_model/bn1/gamma/Read/ReadVariableOp(mnist_model/bn1/beta/Read/ReadVariableOp/mnist_model/bn1/moving_mean/Read/ReadVariableOp3mnist_model/bn1/moving_variance/Read/ReadVariableOp)mnist_model/f1/kernel/Read/ReadVariableOp'mnist_model/f1/bias/Read/ReadVariableOp)mnist_model/f2/kernel/Read/ReadVariableOp'mnist_model/f2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/mnist_model/conv1/kernel/m/Read/ReadVariableOp1Adam/mnist_model/conv1/bias/m/Read/ReadVariableOp0Adam/mnist_model/bn1/gamma/m/Read/ReadVariableOp/Adam/mnist_model/bn1/beta/m/Read/ReadVariableOp0Adam/mnist_model/f1/kernel/m/Read/ReadVariableOp.Adam/mnist_model/f1/bias/m/Read/ReadVariableOp0Adam/mnist_model/f2/kernel/m/Read/ReadVariableOp.Adam/mnist_model/f2/bias/m/Read/ReadVariableOp3Adam/mnist_model/conv1/kernel/v/Read/ReadVariableOp1Adam/mnist_model/conv1/bias/v/Read/ReadVariableOp0Adam/mnist_model/bn1/gamma/v/Read/ReadVariableOp/Adam/mnist_model/bn1/beta/v/Read/ReadVariableOp0Adam/mnist_model/f1/kernel/v/Read/ReadVariableOp.Adam/mnist_model/f1/bias/v/Read/ReadVariableOp0Adam/mnist_model/f2/kernel/v/Read/ReadVariableOp.Adam/mnist_model/f2/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_24502
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemnist_model/conv1/kernelmnist_model/conv1/biasmnist_model/bn1/gammamnist_model/bn1/betamnist_model/bn1/moving_meanmnist_model/bn1/moving_variancemnist_model/f1/kernelmnist_model/f1/biasmnist_model/f2/kernelmnist_model/f2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/mnist_model/conv1/kernel/mAdam/mnist_model/conv1/bias/mAdam/mnist_model/bn1/gamma/mAdam/mnist_model/bn1/beta/mAdam/mnist_model/f1/kernel/mAdam/mnist_model/f1/bias/mAdam/mnist_model/f2/kernel/mAdam/mnist_model/f2/bias/mAdam/mnist_model/conv1/kernel/vAdam/mnist_model/conv1/bias/vAdam/mnist_model/bn1/gamma/vAdam/mnist_model/bn1/beta/vAdam/mnist_model/f1/kernel/vAdam/mnist_model/f1/bias/vAdam/mnist_model/f2/kernel/vAdam/mnist_model/f2/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_24617??
?
_
C__inference_maxpool1_layer_call_and_return_conditional_losses_23367

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_mnist_model_layer_call_fn_23933
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_mnist_model_layer_call_and_return_conditional_losses_236882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
#__inference_bn1_layer_call_fn_24259

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_233502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_maxpool1_layer_call_fn_23373

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_233672
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_activation1_layer_call_fn_24269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation1_layer_call_and_return_conditional_losses_234812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_24233

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?:
?
 __inference__wrapped_model_23257
input_14
0mnist_model_conv1_conv2d_readvariableop_resource5
1mnist_model_conv1_biasadd_readvariableop_resource+
'mnist_model_bn1_readvariableop_resource-
)mnist_model_bn1_readvariableop_1_resource<
8mnist_model_bn1_fusedbatchnormv3_readvariableop_resource>
:mnist_model_bn1_fusedbatchnormv3_readvariableop_1_resource1
-mnist_model_f1_matmul_readvariableop_resource2
.mnist_model_f1_biasadd_readvariableop_resource1
-mnist_model_f2_matmul_readvariableop_resource2
.mnist_model_f2_biasadd_readvariableop_resource
identity??/mnist_model/bn1/FusedBatchNormV3/ReadVariableOp?1mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_1?mnist_model/bn1/ReadVariableOp? mnist_model/bn1/ReadVariableOp_1?(mnist_model/conv1/BiasAdd/ReadVariableOp?'mnist_model/conv1/Conv2D/ReadVariableOp?%mnist_model/f1/BiasAdd/ReadVariableOp?$mnist_model/f1/MatMul/ReadVariableOp?%mnist_model/f2/BiasAdd/ReadVariableOp?$mnist_model/f2/MatMul/ReadVariableOp?
'mnist_model/conv1/Conv2D/ReadVariableOpReadVariableOp0mnist_model_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'mnist_model/conv1/Conv2D/ReadVariableOp?
mnist_model/conv1/Conv2DConv2Dinput_1/mnist_model/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
mnist_model/conv1/Conv2D?
(mnist_model/conv1/BiasAdd/ReadVariableOpReadVariableOp1mnist_model_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(mnist_model/conv1/BiasAdd/ReadVariableOp?
mnist_model/conv1/BiasAddBiasAdd!mnist_model/conv1/Conv2D:output:00mnist_model/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
mnist_model/conv1/BiasAdd?
mnist_model/bn1/ReadVariableOpReadVariableOp'mnist_model_bn1_readvariableop_resource*
_output_shapes
:*
dtype02 
mnist_model/bn1/ReadVariableOp?
 mnist_model/bn1/ReadVariableOp_1ReadVariableOp)mnist_model_bn1_readvariableop_1_resource*
_output_shapes
:*
dtype02"
 mnist_model/bn1/ReadVariableOp_1?
/mnist_model/bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp8mnist_model_bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype021
/mnist_model/bn1/FusedBatchNormV3/ReadVariableOp?
1mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp:mnist_model_bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype023
1mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_1?
 mnist_model/bn1/FusedBatchNormV3FusedBatchNormV3"mnist_model/conv1/BiasAdd:output:0&mnist_model/bn1/ReadVariableOp:value:0(mnist_model/bn1/ReadVariableOp_1:value:07mnist_model/bn1/FusedBatchNormV3/ReadVariableOp:value:09mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2"
 mnist_model/bn1/FusedBatchNormV3?
mnist_model/activation1/ReluRelu$mnist_model/bn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
mnist_model/activation1/Relu?
mnist_model/maxpool1/MaxPoolMaxPool*mnist_model/activation1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
mnist_model/maxpool1/MaxPool?
mnist_model/drop1/IdentityIdentity%mnist_model/maxpool1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
mnist_model/drop1/Identity?
mnist_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
mnist_model/flatten/Const?
mnist_model/flatten/ReshapeReshape#mnist_model/drop1/Identity:output:0"mnist_model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
mnist_model/flatten/Reshape?
$mnist_model/f1/MatMul/ReadVariableOpReadVariableOp-mnist_model_f1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02&
$mnist_model/f1/MatMul/ReadVariableOp?
mnist_model/f1/MatMulMatMul$mnist_model/flatten/Reshape:output:0,mnist_model/f1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mnist_model/f1/MatMul?
%mnist_model/f1/BiasAdd/ReadVariableOpReadVariableOp.mnist_model_f1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%mnist_model/f1/BiasAdd/ReadVariableOp?
mnist_model/f1/BiasAddBiasAddmnist_model/f1/MatMul:product:0-mnist_model/f1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
mnist_model/f1/BiasAdd?
mnist_model/f1/ReluRelumnist_model/f1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
mnist_model/f1/Relu?
mnist_model/dropout/IdentityIdentity!mnist_model/f1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
mnist_model/dropout/Identity?
$mnist_model/f2/MatMul/ReadVariableOpReadVariableOp-mnist_model_f2_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02&
$mnist_model/f2/MatMul/ReadVariableOp?
mnist_model/f2/MatMulMatMul%mnist_model/dropout/Identity:output:0,mnist_model/f2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
mnist_model/f2/MatMul?
%mnist_model/f2/BiasAdd/ReadVariableOpReadVariableOp.mnist_model_f2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%mnist_model/f2/BiasAdd/ReadVariableOp?
mnist_model/f2/BiasAddBiasAddmnist_model/f2/MatMul:product:0-mnist_model/f2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
mnist_model/f2/BiasAdd?
mnist_model/f2/SoftmaxSoftmaxmnist_model/f2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
mnist_model/f2/Softmax?
IdentityIdentity mnist_model/f2/Softmax:softmax:00^mnist_model/bn1/FusedBatchNormV3/ReadVariableOp2^mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_1^mnist_model/bn1/ReadVariableOp!^mnist_model/bn1/ReadVariableOp_1)^mnist_model/conv1/BiasAdd/ReadVariableOp(^mnist_model/conv1/Conv2D/ReadVariableOp&^mnist_model/f1/BiasAdd/ReadVariableOp%^mnist_model/f1/MatMul/ReadVariableOp&^mnist_model/f2/BiasAdd/ReadVariableOp%^mnist_model/f2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2b
/mnist_model/bn1/FusedBatchNormV3/ReadVariableOp/mnist_model/bn1/FusedBatchNormV3/ReadVariableOp2f
1mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_11mnist_model/bn1/FusedBatchNormV3/ReadVariableOp_12@
mnist_model/bn1/ReadVariableOpmnist_model/bn1/ReadVariableOp2D
 mnist_model/bn1/ReadVariableOp_1 mnist_model/bn1/ReadVariableOp_12T
(mnist_model/conv1/BiasAdd/ReadVariableOp(mnist_model/conv1/BiasAdd/ReadVariableOp2R
'mnist_model/conv1/Conv2D/ReadVariableOp'mnist_model/conv1/Conv2D/ReadVariableOp2N
%mnist_model/f1/BiasAdd/ReadVariableOp%mnist_model/f1/BiasAdd/ReadVariableOp2L
$mnist_model/f1/MatMul/ReadVariableOp$mnist_model/f1/MatMul/ReadVariableOp2N
%mnist_model/f2/BiasAdd/ReadVariableOp%mnist_model/f2/BiasAdd/ReadVariableOp2L
$mnist_model/f2/MatMul/ReadVariableOp$mnist_model/f2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
^
@__inference_drop1_layer_call_and_return_conditional_losses_23507

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
@__inference_drop1_layer_call_and_return_conditional_losses_24281

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_24354

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_235782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_23688
x
conv1_23658
conv1_23660
	bn1_23663
	bn1_23665
	bn1_23667
	bn1_23669
f1_23676
f1_23678
f2_23682
f2_23684
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?drop1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?f1/StatefulPartitionedCall?f2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallxconv1_23658conv1_23660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_233872
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_23663	bn1_23665	bn1_23667	bn1_23669*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_234222
bn1/StatefulPartitionedCall?
activation1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation1_layer_call_and_return_conditional_losses_234812
activation1/PartitionedCall?
maxpool1/PartitionedCallPartitionedCall$activation1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_233672
maxpool1/PartitionedCall?
drop1/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_drop1_layer_call_and_return_conditional_losses_235022
drop1/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&drop1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_235262
flatten/PartitionedCall?
f1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0f1_23676f1_23678*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f1_layer_call_and_return_conditional_losses_235452
f1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall#f1/StatefulPartitionedCall:output:0^drop1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_235732!
dropout/StatefulPartitionedCall?
f2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0f2_23682f2_23684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f2_layer_call_and_return_conditional_losses_236022
f2/StatefulPartitionedCall?
IdentityIdentity#f2/StatefulPartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^drop1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^f1/StatefulPartitionedCall^f2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
drop1/StatefulPartitionedCalldrop1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall28
f1/StatefulPartitionedCallf1/StatefulPartitionedCall28
f2/StatefulPartitionedCallf2/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
=__inference_f2_layer_call_and_return_conditional_losses_24365

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_24339

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_23578

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_24302

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_24182

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_234222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_24018
x(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource%
!f1_matmul_readvariableop_resource&
"f1_biasadd_readvariableop_resource%
!f2_matmul_readvariableop_resource&
"f2_biasadd_readvariableop_resource
identity??bn1/AssignNewValue?bn1/AssignNewValue_1?#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?f1/BiasAdd/ReadVariableOp?f1/MatMul/ReadVariableOp?f2/BiasAdd/ReadVariableOp?f2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dx#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAdd?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1/FusedBatchNormV3?
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue?
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue_1?
activation1/ReluRelubn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation1/Relu?
maxpool1/MaxPoolMaxPoolactivation1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPoolo
drop1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
drop1/dropout/Const?
drop1/dropout/MulMulmaxpool1/MaxPool:output:0drop1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/Muls
drop1/dropout/ShapeShapemaxpool1/MaxPool:output:0*
T0*
_output_shapes
:2
drop1/dropout/Shape?
*drop1/dropout/random_uniform/RandomUniformRandomUniformdrop1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02,
*drop1/dropout/random_uniform/RandomUniform?
drop1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
drop1/dropout/GreaterEqual/y?
drop1/dropout/GreaterEqualGreaterEqual3drop1/dropout/random_uniform/RandomUniform:output:0%drop1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/GreaterEqual?
drop1/dropout/CastCastdrop1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
drop1/dropout/Cast?
drop1/dropout/Mul_1Muldrop1/dropout/Mul:z:0drop1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedrop1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
f1/MatMul/ReadVariableOpReadVariableOp!f1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
f1/MatMul/ReadVariableOp?
	f1/MatMulMatMulflatten/Reshape:output:0 f1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	f1/MatMul?
f1/BiasAdd/ReadVariableOpReadVariableOp"f1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
f1/BiasAdd/ReadVariableOp?

f1/BiasAddBiasAddf1/MatMul:product:0!f1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

f1/BiasAddb
f1/ReluReluf1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
f1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulf1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Muls
dropout/dropout/ShapeShapef1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
f2/MatMul/ReadVariableOpReadVariableOp!f2_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
f2/MatMul/ReadVariableOp?
	f2/MatMulMatMuldropout/dropout/Mul_1:z:0 f2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	f2/MatMul?
f2/BiasAdd/ReadVariableOpReadVariableOp"f2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
f2/BiasAdd/ReadVariableOp?

f2/BiasAddBiasAddf2/MatMul:product:0!f2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

f2/BiasAddj

f2/SoftmaxSoftmaxf2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

f2/Softmax?
IdentityIdentityf2/Softmax:softmax:0^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^f1/BiasAdd/ReadVariableOp^f1/MatMul/ReadVariableOp^f2/BiasAdd/ReadVariableOp^f2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2(
bn1/AssignNewValuebn1/AssignNewValue2,
bn1/AssignNewValue_1bn1/AssignNewValue_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp26
f1/BiasAdd/ReadVariableOpf1/BiasAdd/ReadVariableOp24
f1/MatMul/ReadVariableOpf1/MatMul/ReadVariableOp26
f2/BiasAdd/ReadVariableOpf2/BiasAdd/ReadVariableOp24
f2/MatMul/ReadVariableOpf2/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
w
"__inference_f2_layer_call_fn_24374

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f2_layer_call_and_return_conditional_losses_236022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
@__inference_drop1_layer_call_and_return_conditional_losses_23502

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_24349

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_235732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_activation1_layer_call_and_return_conditional_losses_24264

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_23526

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_24617
file_prefix-
)assignvariableop_mnist_model_conv1_kernel-
)assignvariableop_1_mnist_model_conv1_bias,
(assignvariableop_2_mnist_model_bn1_gamma+
'assignvariableop_3_mnist_model_bn1_beta2
.assignvariableop_4_mnist_model_bn1_moving_mean6
2assignvariableop_5_mnist_model_bn1_moving_variance,
(assignvariableop_6_mnist_model_f1_kernel*
&assignvariableop_7_mnist_model_f1_bias,
(assignvariableop_8_mnist_model_f2_kernel*
&assignvariableop_9_mnist_model_f2_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_17
3assignvariableop_19_adam_mnist_model_conv1_kernel_m5
1assignvariableop_20_adam_mnist_model_conv1_bias_m4
0assignvariableop_21_adam_mnist_model_bn1_gamma_m3
/assignvariableop_22_adam_mnist_model_bn1_beta_m4
0assignvariableop_23_adam_mnist_model_f1_kernel_m2
.assignvariableop_24_adam_mnist_model_f1_bias_m4
0assignvariableop_25_adam_mnist_model_f2_kernel_m2
.assignvariableop_26_adam_mnist_model_f2_bias_m7
3assignvariableop_27_adam_mnist_model_conv1_kernel_v5
1assignvariableop_28_adam_mnist_model_conv1_bias_v4
0assignvariableop_29_adam_mnist_model_bn1_gamma_v3
/assignvariableop_30_adam_mnist_model_bn1_beta_v4
0assignvariableop_31_adam_mnist_model_f1_kernel_v2
.assignvariableop_32_adam_mnist_model_f1_bias_v4
0assignvariableop_33_adam_mnist_model_f2_kernel_v2
.assignvariableop_34_adam_mnist_model_f2_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB#b1/gamma/.ATTRIBUTES/VARIABLE_VALUEB"b1/beta/.ATTRIBUTES/VARIABLE_VALUEB)b1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-b1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$f1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f1/bias/.ATTRIBUTES/VARIABLE_VALUEB$f2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?b1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>b1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?b1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>b1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_mnist_model_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_mnist_model_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_mnist_model_bn1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp'assignvariableop_3_mnist_model_bn1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_mnist_model_bn1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp2assignvariableop_5_mnist_model_bn1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_mnist_model_f1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_mnist_model_f1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_mnist_model_f2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_mnist_model_f2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_mnist_model_conv1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_mnist_model_conv1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_mnist_model_bn1_gamma_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_adam_mnist_model_bn1_beta_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_mnist_model_f1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_mnist_model_f1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_mnist_model_f2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_mnist_model_f2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp3assignvariableop_27_adam_mnist_model_conv1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_mnist_model_conv1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_mnist_model_bn1_gamma_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_mnist_model_bn1_beta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_mnist_model_f1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_mnist_model_f1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_mnist_model_f2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_mnist_model_f2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_23350

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
@__inference_conv1_layer_call_and_return_conditional_losses_23387

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_24062
x(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource%
!f1_matmul_readvariableop_resource&
"f1_biasadd_readvariableop_resource%
!f2_matmul_readvariableop_resource&
"f2_biasadd_readvariableop_resource
identity??#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?f1/BiasAdd/ReadVariableOp?f1/MatMul/ReadVariableOp?f2/BiasAdd/ReadVariableOp?f2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dx#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAdd?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
bn1/FusedBatchNormV3?
activation1/ReluRelubn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation1/Relu?
maxpool1/MaxPoolMaxPoolactivation1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool?
drop1/IdentityIdentitymaxpool1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
drop1/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedrop1/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
f1/MatMul/ReadVariableOpReadVariableOp!f1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
f1/MatMul/ReadVariableOp?
	f1/MatMulMatMulflatten/Reshape:output:0 f1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	f1/MatMul?
f1/BiasAdd/ReadVariableOpReadVariableOp"f1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
f1/BiasAdd/ReadVariableOp?

f1/BiasAddBiasAddf1/MatMul:product:0!f1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

f1/BiasAddb
f1/ReluReluf1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
f1/Reluz
dropout/IdentityIdentityf1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
f2/MatMul/ReadVariableOpReadVariableOp!f2_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
f2/MatMul/ReadVariableOp?
	f2/MatMulMatMuldropout/Identity:output:0 f2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	f2/MatMul?
f2/BiasAdd/ReadVariableOpReadVariableOp"f2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
f2/BiasAdd/ReadVariableOp?

f2/BiasAddBiasAddf2/MatMul:product:0!f2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

f2/BiasAddj

f2/SoftmaxSoftmaxf2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

f2/Softmax?
IdentityIdentityf2/Softmax:softmax:0$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^f1/BiasAdd/ReadVariableOp^f1/MatMul/ReadVariableOp^f2/BiasAdd/ReadVariableOp^f2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp26
f1/BiasAdd/ReadVariableOpf1/BiasAdd/ReadVariableOp24
f1/MatMul/ReadVariableOpf1/MatMul/ReadVariableOp26
f2/BiasAdd/ReadVariableOpf2/BiasAdd/ReadVariableOp24
f2/MatMul/ReadVariableOpf2/MatMul/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
#__inference_signature_wrapper_23804
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_232572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
+__inference_mnist_model_layer_call_fn_23958
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_mnist_model_layer_call_and_return_conditional_losses_237462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
w
"__inference_f1_layer_call_fn_24327

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f1_layer_call_and_return_conditional_losses_235452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_24307

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_235262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_24246

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_233192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
=__inference_f1_layer_call_and_return_conditional_losses_24318

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
b
F__inference_activation1_layer_call_and_return_conditional_losses_23481

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_bn1_layer_call_fn_24195

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_234402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
A
%__inference_drop1_layer_call_fn_24296

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_drop1_layer_call_and_return_conditional_losses_235072
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
z
%__inference_conv1_layer_call_fn_24131

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_233872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
=__inference_f2_layer_call_and_return_conditional_losses_23602

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_mnist_model_layer_call_fn_24087
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_mnist_model_layer_call_and_return_conditional_losses_236882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
^
@__inference_drop1_layer_call_and_return_conditional_losses_24286

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_23440

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_24169

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
%__inference_drop1_layer_call_fn_24291

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_drop1_layer_call_and_return_conditional_losses_235022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
=__inference_f1_layer_call_and_return_conditional_losses_23545

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
+__inference_mnist_model_layer_call_fn_24112
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_mnist_model_layer_call_and_return_conditional_losses_237462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?.
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_23908
input_1(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource%
!f1_matmul_readvariableop_resource&
"f1_biasadd_readvariableop_resource%
!f2_matmul_readvariableop_resource&
"f2_biasadd_readvariableop_resource
identity??#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?f1/BiasAdd/ReadVariableOp?f1/MatMul/ReadVariableOp?f2/BiasAdd/ReadVariableOp?f2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinput_1#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAdd?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
bn1/FusedBatchNormV3?
activation1/ReluRelubn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation1/Relu?
maxpool1/MaxPoolMaxPoolactivation1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool?
drop1/IdentityIdentitymaxpool1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
drop1/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedrop1/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
f1/MatMul/ReadVariableOpReadVariableOp!f1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
f1/MatMul/ReadVariableOp?
	f1/MatMulMatMulflatten/Reshape:output:0 f1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	f1/MatMul?
f1/BiasAdd/ReadVariableOpReadVariableOp"f1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
f1/BiasAdd/ReadVariableOp?

f1/BiasAddBiasAddf1/MatMul:product:0!f1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

f1/BiasAddb
f1/ReluReluf1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
f1/Reluz
dropout/IdentityIdentityf1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
f2/MatMul/ReadVariableOpReadVariableOp!f2_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
f2/MatMul/ReadVariableOp?
	f2/MatMulMatMuldropout/Identity:output:0 f2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	f2/MatMul?
f2/BiasAdd/ReadVariableOpReadVariableOp"f2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
f2/BiasAdd/ReadVariableOp?

f2/BiasAddBiasAddf2/MatMul:product:0!f2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

f2/BiasAddj

f2/SoftmaxSoftmaxf2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

f2/Softmax?
IdentityIdentityf2/Softmax:softmax:0$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^f1/BiasAdd/ReadVariableOp^f1/MatMul/ReadVariableOp^f2/BiasAdd/ReadVariableOp^f2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp26
f1/BiasAdd/ReadVariableOpf1/BiasAdd/ReadVariableOp24
f1/MatMul/ReadVariableOpf1/MatMul/ReadVariableOp26
f2/BiasAdd/ReadVariableOpf2/BiasAdd/ReadVariableOp24
f2/MatMul/ReadVariableOpf2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_24215

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_24151

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?J
?
__inference__traced_save_24502
file_prefix7
3savev2_mnist_model_conv1_kernel_read_readvariableop5
1savev2_mnist_model_conv1_bias_read_readvariableop4
0savev2_mnist_model_bn1_gamma_read_readvariableop3
/savev2_mnist_model_bn1_beta_read_readvariableop:
6savev2_mnist_model_bn1_moving_mean_read_readvariableop>
:savev2_mnist_model_bn1_moving_variance_read_readvariableop4
0savev2_mnist_model_f1_kernel_read_readvariableop2
.savev2_mnist_model_f1_bias_read_readvariableop4
0savev2_mnist_model_f2_kernel_read_readvariableop2
.savev2_mnist_model_f2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_mnist_model_conv1_kernel_m_read_readvariableop<
8savev2_adam_mnist_model_conv1_bias_m_read_readvariableop;
7savev2_adam_mnist_model_bn1_gamma_m_read_readvariableop:
6savev2_adam_mnist_model_bn1_beta_m_read_readvariableop;
7savev2_adam_mnist_model_f1_kernel_m_read_readvariableop9
5savev2_adam_mnist_model_f1_bias_m_read_readvariableop;
7savev2_adam_mnist_model_f2_kernel_m_read_readvariableop9
5savev2_adam_mnist_model_f2_bias_m_read_readvariableop>
:savev2_adam_mnist_model_conv1_kernel_v_read_readvariableop<
8savev2_adam_mnist_model_conv1_bias_v_read_readvariableop;
7savev2_adam_mnist_model_bn1_gamma_v_read_readvariableop:
6savev2_adam_mnist_model_bn1_beta_v_read_readvariableop;
7savev2_adam_mnist_model_f1_kernel_v_read_readvariableop9
5savev2_adam_mnist_model_f1_bias_v_read_readvariableop;
7savev2_adam_mnist_model_f2_kernel_v_read_readvariableop9
5savev2_adam_mnist_model_f2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB#b1/gamma/.ATTRIBUTES/VARIABLE_VALUEB"b1/beta/.ATTRIBUTES/VARIABLE_VALUEB)b1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-b1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$f1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f1/bias/.ATTRIBUTES/VARIABLE_VALUEB$f2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?b1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>b1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?b1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>b1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_mnist_model_conv1_kernel_read_readvariableop1savev2_mnist_model_conv1_bias_read_readvariableop0savev2_mnist_model_bn1_gamma_read_readvariableop/savev2_mnist_model_bn1_beta_read_readvariableop6savev2_mnist_model_bn1_moving_mean_read_readvariableop:savev2_mnist_model_bn1_moving_variance_read_readvariableop0savev2_mnist_model_f1_kernel_read_readvariableop.savev2_mnist_model_f1_bias_read_readvariableop0savev2_mnist_model_f2_kernel_read_readvariableop.savev2_mnist_model_f2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_mnist_model_conv1_kernel_m_read_readvariableop8savev2_adam_mnist_model_conv1_bias_m_read_readvariableop7savev2_adam_mnist_model_bn1_gamma_m_read_readvariableop6savev2_adam_mnist_model_bn1_beta_m_read_readvariableop7savev2_adam_mnist_model_f1_kernel_m_read_readvariableop5savev2_adam_mnist_model_f1_bias_m_read_readvariableop7savev2_adam_mnist_model_f2_kernel_m_read_readvariableop5savev2_adam_mnist_model_f2_bias_m_read_readvariableop:savev2_adam_mnist_model_conv1_kernel_v_read_readvariableop8savev2_adam_mnist_model_conv1_bias_v_read_readvariableop7savev2_adam_mnist_model_bn1_gamma_v_read_readvariableop6savev2_adam_mnist_model_bn1_beta_v_read_readvariableop7savev2_adam_mnist_model_f1_kernel_v_read_readvariableop5savev2_adam_mnist_model_f1_bias_v_read_readvariableop7savev2_adam_mnist_model_f2_kernel_v_read_readvariableop5savev2_adam_mnist_model_f2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::
?	?:?:	?
:
: : : : : : : : : :::::
?	?:?:	?
:
:::::
?	?:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:%	!

_output_shapes
:	?
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::& "
 
_output_shapes
:
?	?:!!

_output_shapes	
:?:%"!

_output_shapes
:	?
: #

_output_shapes
:
:$

_output_shapes
: 
?#
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_23746
x
conv1_23716
conv1_23718
	bn1_23721
	bn1_23723
	bn1_23725
	bn1_23727
f1_23734
f1_23736
f2_23740
f2_23742
identity??bn1/StatefulPartitionedCall?conv1/StatefulPartitionedCall?f1/StatefulPartitionedCall?f2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallxconv1_23716conv1_23718*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1_layer_call_and_return_conditional_losses_233872
conv1/StatefulPartitionedCall?
bn1/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	bn1_23721	bn1_23723	bn1_23725	bn1_23727*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_bn1_layer_call_and_return_conditional_losses_234402
bn1/StatefulPartitionedCall?
activation1/PartitionedCallPartitionedCall$bn1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_activation1_layer_call_and_return_conditional_losses_234812
activation1/PartitionedCall?
maxpool1/PartitionedCallPartitionedCall$activation1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_233672
maxpool1/PartitionedCall?
drop1/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_drop1_layer_call_and_return_conditional_losses_235072
drop1/PartitionedCall?
flatten/PartitionedCallPartitionedCalldrop1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_235262
flatten/PartitionedCall?
f1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0f1_23734f1_23736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f1_layer_call_and_return_conditional_losses_235452
f1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall#f1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_235782
dropout/PartitionedCall?
f2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0f2_23740f2_23742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_f2_layer_call_and_return_conditional_losses_236022
f2/StatefulPartitionedCall?
IdentityIdentity#f2/StatefulPartitionedCall:output:0^bn1/StatefulPartitionedCall^conv1/StatefulPartitionedCall^f1/StatefulPartitionedCall^f2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2:
bn1/StatefulPartitionedCallbn1/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall28
f1/StatefulPartitionedCallf1/StatefulPartitionedCall28
f2/StatefulPartitionedCallf2/StatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?	
?
@__inference_conv1_layer_call_and_return_conditional_losses_24122

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_23319

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_24344

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_bn1_layer_call_and_return_conditional_losses_23422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?G
?
F__inference_mnist_model_layer_call_and_return_conditional_losses_23864
input_1(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource
bn1_readvariableop_resource!
bn1_readvariableop_1_resource0
,bn1_fusedbatchnormv3_readvariableop_resource2
.bn1_fusedbatchnormv3_readvariableop_1_resource%
!f1_matmul_readvariableop_resource&
"f1_biasadd_readvariableop_resource%
!f2_matmul_readvariableop_resource&
"f2_biasadd_readvariableop_resource
identity??bn1/AssignNewValue?bn1/AssignNewValue_1?#bn1/FusedBatchNormV3/ReadVariableOp?%bn1/FusedBatchNormV3/ReadVariableOp_1?bn1/ReadVariableOp?bn1/ReadVariableOp_1?conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?f1/BiasAdd/ReadVariableOp?f1/MatMul/ReadVariableOp?f2/BiasAdd/ReadVariableOp?f2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinput_1#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAdd?
bn1/ReadVariableOpReadVariableOpbn1_readvariableop_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp?
bn1/ReadVariableOp_1ReadVariableOpbn1_readvariableop_1_resource*
_output_shapes
:*
dtype02
bn1/ReadVariableOp_1?
#bn1/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02%
#bn1/FusedBatchNormV3/ReadVariableOp?
%bn1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%bn1/FusedBatchNormV3/ReadVariableOp_1?
bn1/FusedBatchNormV3FusedBatchNormV3conv1/BiasAdd:output:0bn1/ReadVariableOp:value:0bn1/ReadVariableOp_1:value:0+bn1/FusedBatchNormV3/ReadVariableOp:value:0-bn1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
bn1/FusedBatchNormV3?
bn1/AssignNewValueAssignVariableOp,bn1_fusedbatchnormv3_readvariableop_resource!bn1/FusedBatchNormV3:batch_mean:0$^bn1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*?
_class5
31loc:@bn1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue?
bn1/AssignNewValue_1AssignVariableOp.bn1_fusedbatchnormv3_readvariableop_1_resource%bn1/FusedBatchNormV3:batch_variance:0&^bn1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*A
_class7
53loc:@bn1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
bn1/AssignNewValue_1?
activation1/ReluRelubn1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation1/Relu?
maxpool1/MaxPoolMaxPoolactivation1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPoolo
drop1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
drop1/dropout/Const?
drop1/dropout/MulMulmaxpool1/MaxPool:output:0drop1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/Muls
drop1/dropout/ShapeShapemaxpool1/MaxPool:output:0*
T0*
_output_shapes
:2
drop1/dropout/Shape?
*drop1/dropout/random_uniform/RandomUniformRandomUniformdrop1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02,
*drop1/dropout/random_uniform/RandomUniform?
drop1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
drop1/dropout/GreaterEqual/y?
drop1/dropout/GreaterEqualGreaterEqual3drop1/dropout/random_uniform/RandomUniform:output:0%drop1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/GreaterEqual?
drop1/dropout/CastCastdrop1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
drop1/dropout/Cast?
drop1/dropout/Mul_1Muldrop1/dropout/Mul:z:0drop1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
drop1/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshapedrop1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
f1/MatMul/ReadVariableOpReadVariableOp!f1_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
f1/MatMul/ReadVariableOp?
	f1/MatMulMatMulflatten/Reshape:output:0 f1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	f1/MatMul?
f1/BiasAdd/ReadVariableOpReadVariableOp"f1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
f1/BiasAdd/ReadVariableOp?

f1/BiasAddBiasAddf1/MatMul:product:0!f1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

f1/BiasAddb
f1/ReluReluf1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
f1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulf1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Muls
dropout/dropout/ShapeShapef1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
f2/MatMul/ReadVariableOpReadVariableOp!f2_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
f2/MatMul/ReadVariableOp?
	f2/MatMulMatMuldropout/dropout/Mul_1:z:0 f2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
	f2/MatMul?
f2/BiasAdd/ReadVariableOpReadVariableOp"f2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
f2/BiasAdd/ReadVariableOp?

f2/BiasAddBiasAddf2/MatMul:product:0!f2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2

f2/BiasAddj

f2/SoftmaxSoftmaxf2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2

f2/Softmax?
IdentityIdentityf2/Softmax:softmax:0^bn1/AssignNewValue^bn1/AssignNewValue_1$^bn1/FusedBatchNormV3/ReadVariableOp&^bn1/FusedBatchNormV3/ReadVariableOp_1^bn1/ReadVariableOp^bn1/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^f1/BiasAdd/ReadVariableOp^f1/MatMul/ReadVariableOp^f2/BiasAdd/ReadVariableOp^f2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2(
bn1/AssignNewValuebn1/AssignNewValue2,
bn1/AssignNewValue_1bn1/AssignNewValue_12J
#bn1/FusedBatchNormV3/ReadVariableOp#bn1/FusedBatchNormV3/ReadVariableOp2N
%bn1/FusedBatchNormV3/ReadVariableOp_1%bn1/FusedBatchNormV3/ReadVariableOp_12(
bn1/ReadVariableOpbn1/ReadVariableOp2,
bn1/ReadVariableOp_1bn1/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp26
f1/BiasAdd/ReadVariableOpf1/BiasAdd/ReadVariableOp24
f1/MatMul/ReadVariableOpf1/MatMul/ReadVariableOp26
f2/BiasAdd/ReadVariableOpf2/BiasAdd/ReadVariableOp24
f2/MatMul/ReadVariableOpf2/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_23573

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?	
c1
b1
a1
p1
d1
flatten
f1
d2
	f2

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "MnistModel", "name": "mnist_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MnistModel"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 28, 28, 1]}}
?	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "bn1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bn1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 28, 28, 6]}}
?
regularization_losses
 	variables
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "drop1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "drop1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "f1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "f1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1176}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1176]}}
?
5regularization_losses
6	variables
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "f2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "f2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 128]}}
?
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_ratem?m?m?m?/m?0m?9m?:m?v?v?v?v?/v?0v?9v?:v?"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
/6
07
98
:9"
trackable_list_wrapper
X
0
1
2
3
/4
05
96
:7"
trackable_list_wrapper
?

Dlayers
regularization_losses
Enon_trainable_variables
Flayer_regularization_losses
	variables
Gmetrics
trainable_variables
Hlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
2:02mnist_model/conv1/kernel
$:"2mnist_model/conv1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ilayers
regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses
	variables
Lmetrics
trainable_variables
Mlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
#:!2mnist_model/bn1/gamma
": 2mnist_model/bn1/beta
+:) (2mnist_model/bn1/moving_mean
/:- (2mnist_model/bn1/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Nlayers
regularization_losses
Onon_trainable_variables
Player_regularization_losses
	variables
Qmetrics
trainable_variables
Rlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Slayers
regularization_losses
Tnon_trainable_variables
Ulayer_regularization_losses
 	variables
Vmetrics
!trainable_variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Xlayers
#regularization_losses
Ynon_trainable_variables
Zlayer_regularization_losses
$	variables
[metrics
%trainable_variables
\layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

]layers
'regularization_losses
^non_trainable_variables
_layer_regularization_losses
(	variables
`metrics
)trainable_variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

blayers
+regularization_losses
cnon_trainable_variables
dlayer_regularization_losses
,	variables
emetrics
-trainable_variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
?	?2mnist_model/f1/kernel
": ?2mnist_model/f1/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?

glayers
1regularization_losses
hnon_trainable_variables
ilayer_regularization_losses
2	variables
jmetrics
3trainable_variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

llayers
5regularization_losses
mnon_trainable_variables
nlayer_regularization_losses
6	variables
ometrics
7trainable_variables
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&	?
2mnist_model/f2/kernel
!:
2mnist_model/f2/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?

qlayers
;regularization_losses
rnon_trainable_variables
slayer_regularization_losses
<	variables
tmetrics
=trainable_variables
ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	xtotal
	ycount
z	variables
{	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	|total
	}count
~
_fn_kwargs
	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
-
	variables"
_generic_user_object
7:52Adam/mnist_model/conv1/kernel/m
):'2Adam/mnist_model/conv1/bias/m
(:&2Adam/mnist_model/bn1/gamma/m
':%2Adam/mnist_model/bn1/beta/m
.:,
?	?2Adam/mnist_model/f1/kernel/m
':%?2Adam/mnist_model/f1/bias/m
-:+	?
2Adam/mnist_model/f2/kernel/m
&:$
2Adam/mnist_model/f2/bias/m
7:52Adam/mnist_model/conv1/kernel/v
):'2Adam/mnist_model/conv1/bias/v
(:&2Adam/mnist_model/bn1/gamma/v
':%2Adam/mnist_model/bn1/beta/v
.:,
?	?2Adam/mnist_model/f1/kernel/v
':%?2Adam/mnist_model/f1/bias/v
-:+	?
2Adam/mnist_model/f2/kernel/v
&:$
2Adam/mnist_model/f2/bias/v
?2?
+__inference_mnist_model_layer_call_fn_23958
+__inference_mnist_model_layer_call_fn_23933
+__inference_mnist_model_layer_call_fn_24087
+__inference_mnist_model_layer_call_fn_24112?
???
FullArgSpec,
args$?!
jself
jx

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_23257?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
F__inference_mnist_model_layer_call_and_return_conditional_losses_24062
F__inference_mnist_model_layer_call_and_return_conditional_losses_24018
F__inference_mnist_model_layer_call_and_return_conditional_losses_23908
F__inference_mnist_model_layer_call_and_return_conditional_losses_23864?
???
FullArgSpec,
args$?!
jself
jx

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_conv1_layer_call_fn_24131?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv1_layer_call_and_return_conditional_losses_24122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_bn1_layer_call_fn_24259
#__inference_bn1_layer_call_fn_24195
#__inference_bn1_layer_call_fn_24246
#__inference_bn1_layer_call_fn_24182?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_bn1_layer_call_and_return_conditional_losses_24233
>__inference_bn1_layer_call_and_return_conditional_losses_24151
>__inference_bn1_layer_call_and_return_conditional_losses_24215
>__inference_bn1_layer_call_and_return_conditional_losses_24169?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_activation1_layer_call_fn_24269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation1_layer_call_and_return_conditional_losses_24264?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_maxpool1_layer_call_fn_23373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_maxpool1_layer_call_and_return_conditional_losses_23367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
%__inference_drop1_layer_call_fn_24296
%__inference_drop1_layer_call_fn_24291?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_drop1_layer_call_and_return_conditional_losses_24286
@__inference_drop1_layer_call_and_return_conditional_losses_24281?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_24307?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_24302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference_f1_layer_call_fn_24327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_f1_layer_call_and_return_conditional_losses_24318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_24349
'__inference_dropout_layer_call_fn_24354?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_24339
B__inference_dropout_layer_call_and_return_conditional_losses_24344?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference_f2_layer_call_fn_24374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_f2_layer_call_and_return_conditional_losses_24365?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_23804input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_23257{
/09:8?5
.?+
)?&
input_1?????????
? "3?0
.
output_1"?
output_1?????????
?
F__inference_activation1_layer_call_and_return_conditional_losses_24264h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_activation1_layer_call_fn_24269[7?4
-?*
(?%
inputs?????????
? " ???????????
>__inference_bn1_layer_call_and_return_conditional_losses_24151r;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_24169r;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_24215?M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
>__inference_bn1_layer_call_and_return_conditional_losses_24233?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
#__inference_bn1_layer_call_fn_24182e;?8
1?.
(?%
inputs?????????
p
? " ???????????
#__inference_bn1_layer_call_fn_24195e;?8
1?.
(?%
inputs?????????
p 
? " ???????????
#__inference_bn1_layer_call_fn_24246?M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
#__inference_bn1_layer_call_fn_24259?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
@__inference_conv1_layer_call_and_return_conditional_losses_24122l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
%__inference_conv1_layer_call_fn_24131_7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_drop1_layer_call_and_return_conditional_losses_24281l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
@__inference_drop1_layer_call_and_return_conditional_losses_24286l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
%__inference_drop1_layer_call_fn_24291_;?8
1?.
(?%
inputs?????????
p
? " ???????????
%__inference_drop1_layer_call_fn_24296_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
B__inference_dropout_layer_call_and_return_conditional_losses_24339^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_24344^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_24349Q4?1
*?'
!?
inputs??????????
p
? "???????????|
'__inference_dropout_layer_call_fn_24354Q4?1
*?'
!?
inputs??????????
p 
? "????????????
=__inference_f1_layer_call_and_return_conditional_losses_24318^/00?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? w
"__inference_f1_layer_call_fn_24327Q/00?-
&?#
!?
inputs??????????	
? "????????????
=__inference_f2_layer_call_and_return_conditional_losses_24365]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? v
"__inference_f2_layer_call_fn_24374P9:0?-
&?#
!?
inputs??????????
? "??????????
?
B__inference_flatten_layer_call_and_return_conditional_losses_24302a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????	
? 
'__inference_flatten_layer_call_fn_24307T7?4
-?*
(?%
inputs?????????
? "???????????	?
C__inference_maxpool1_layer_call_and_return_conditional_losses_23367?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxpool1_layer_call_fn_23373?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_mnist_model_layer_call_and_return_conditional_losses_23864u
/09:@?=
6?3
)?&
input_1?????????
p

 
? "%?"
?
0?????????

? ?
F__inference_mnist_model_layer_call_and_return_conditional_losses_23908u
/09:@?=
6?3
)?&
input_1?????????
p 

 
? "%?"
?
0?????????

? ?
F__inference_mnist_model_layer_call_and_return_conditional_losses_24018o
/09::?7
0?-
#? 
x?????????
p

 
? "%?"
?
0?????????

? ?
F__inference_mnist_model_layer_call_and_return_conditional_losses_24062o
/09::?7
0?-
#? 
x?????????
p 

 
? "%?"
?
0?????????

? ?
+__inference_mnist_model_layer_call_fn_23933h
/09:@?=
6?3
)?&
input_1?????????
p

 
? "??????????
?
+__inference_mnist_model_layer_call_fn_23958h
/09:@?=
6?3
)?&
input_1?????????
p 

 
? "??????????
?
+__inference_mnist_model_layer_call_fn_24087b
/09::?7
0?-
#? 
x?????????
p

 
? "??????????
?
+__inference_mnist_model_layer_call_fn_24112b
/09::?7
0?-
#? 
x?????????
p 

 
? "??????????
?
#__inference_signature_wrapper_23804?
/09:C?@
? 
9?6
4
input_1)?&
input_1?????????"3?0
.
output_1"?
output_1?????????
