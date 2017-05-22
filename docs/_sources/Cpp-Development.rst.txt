C++ Developement
================

If you are interested in creating additional functionality for Bifrost,
this document should help you get up and running with the Bifrost C++
API.

Create a Ring Buffer and Load Data
----------------------------------

.. code:: c

   #include <cuda_runtime_api.h>
   #include <bifrost/common.h>
   #include <bifrost/ring.h>
   
   ...
   
   //////////////////////////////////
   //Create and write to a ring
   /////////////////////////////////
   
   BFring my_ring; //declare our ring variable
   bfRingCreate(&my_ring, BF_SPACE_SYSTEM); //initiate this ring on local memory (=BF_SPACE_SYSTEM)
   bfRingBeginWriting(my_ring); //begin writing to this ring
   //generate some dummy variables
   BFoffset skip_time_tag = -1;
   BFoffset skip_offset = 0;
   BFsize my_header_size = 0;
   char *my_header = "";
   //Set our ring variables
   BFsize nringlets = 1; //dimensionality of our ring.
   BFsize nbytes = 32*8; //number of bytes we are putting in
   BFsize buffer_bytes = 4*nbytes;
   //resize ring to fit the data
   bfRingResize(
       my_ring, nbytes, buffer_bytes, nringlets);
   //open a sequence on the ring.
   BFwsequence my_sequence;
   const char* name = "mysequence"; //we can find our sequence by this name later on
   bfRingSequenceBegin(
       &my_sequence, my_ring, name, skip_time_tag,
       my_header_size, (const void*)my_header, nringlets, 
       skip_offset);
   //reserve a "span" on this sequence to put our data
   BFwspan my_span;
   bfRingSpanReserve(
       &my_span, my_ring, nbytes);
   void *data_access; //create a pointer to pass our data to
   //create the data and copy it to the ring
   float data[8] = {10, -10, 0, 0, 1, -3, 1, 0};
   bfRingSpanGetData((BFspan)my_span, &data_access); //point our pointer to the span's allocated memory 
   memcpy(data_access, &data, nbytes);
   //stop writing
   bfRingSpanCommit(my_span, nbytes); //commit this span to memory
   bfRingSequenceEnd(my_sequence, skip_offset); //end off the sequence
   bfRingEndWriting(my_ring);
   
   //////////////////////////////////
   //Read from the ring
   /////////////////////////////////
   
   nbytes = 32*8; //the size of the data we want to read
   //open the last accessed sequence 
   BFrsequence my_read_sequence;
   bfRingSequenceOpenLatest(
       &my_read_sequence, my_ring, true);
   //open a span on this sequence
   BFrspan my_read_span;
   bfRingSpanAcquire(
       &my_read_span, my_read_sequence, skip_offset, nbytes);
   //Access the data from the span with a pointer
   bfRingSpanGetData((BFspan)my_read_span, &data_access);
   float *my_data = static_cast<float*>(data_access); //Copy the data into a readable format
   //print out the ring data
   for (int i = 0; i < 8; i++)
   {
       printf("%f\n",my_data[i]);
   }
   //close up our ring access
   bfRingSpanRelease(my_read_span);
   bfRingSequenceClose(my_read_sequence);
   bfRingDestroy(my_ring); //delete the ring from memory
