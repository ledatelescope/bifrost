## Contents

1. [Create a Ring Buffer and Load Data](#ringcreation)

## <a name="ringcreation">Create a Ring Buffer and Load Data</a>

```` C++
#include <cuda_runtime_api.h>
#include <bifrost/common.h>
#include <bifrost/ring.h>

...

//declare our ring variable
BFring my_ring;
//initiate this ring on local memory (=BF_SPACE_SYSTEM)
bfRingCreate(my_ring, BF_SPACE_SYSTEM); 
//begin writing to this ring
bfRingBeginWriting(*my_ring);
````