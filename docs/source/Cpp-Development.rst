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

Adding New Packet Formats
------------------------

A wide variety of packet formats are already included in Bifrost. 
For simplicity, it is likely preferable to make use of these pre-existing
formats. In the case that this becomes infeasible, here are some of what 
is necessary in order to add a new format to Bifrost.

Files to edit:

1. ``python/bifrost/packet_capture.py``

   * Add ``set_mypacket`` to the ``PacketCaptureCallback`` class. It will likely look
     very similar to the ``set_chips`` method.

2. ``src/bifrost/packet_capture.h``

   * This is for ctypesgen. Add a typedef for the sequence callback. This typedef
     corresponds to the sequence callback used in the packet reader, see the sections
     on ``test_udp_io.py`` and ``test_disk_io.py`` for examples of writing the packet reader. 
   * Also declare the capture callback. 

3. ``src/formats/format.hpp``

   * Add a one-line ``#include "mypacket.hpp"``

4. `src/formats/mypacket.hpp`

   * This is the only file that will need to be fully written from scratch. The
     easiest way to proceed is to copy the most similar existing packet format and
     modify it accordingly. One will need to make sure that the header is defined
     properly and that the correct information is going into it, and one will need
     to make sure that the `memcpy` operation is properly filling the packet with
     data. 

5. ``src/packet_capture.cpp``

   * Need to add a call to the packet capture callback. 

6. ``src/packet_capture.hpp``

   * This is where you will spend most of your time. Add your packet capture sequence
     callback to the ``BFpacketcapture_callback_impl`` initialization list. Immediately
     after the initialization list, add the ``set_mypacket`` and ``get_mypacket`` methods. 
   * Add a new class: ``BFpacketcapture_mypacket_impl``. In the case of simpler packet
     formats, this may be very close to the already written ``BFpacketcapture_chips_impl``. 
     It's probably best to start by copying the format that is closest to the format
     you are writing and modify it. 
   * In ``BFpacketcapture_create``, add the format to the first long if-else if statement.
     This section tells the disk writer the size of the packet to expect. Then add your
     packet to the second if-else if statement.

7. ``src/packet_writer.hpp``

   * After the ``BFpacketwriter_generic_impl``, add a class ``BFpacketwriter_mypacket_impl``.
     Take care to choose the correct BF\_DTYPE\_???. 
   * In ``BFpacketwriter_create``, add your packet to the first if-else if statement.
     Note that nsamples needs to correspond to the number elements in the data portion
     of the packet. Then add your packet to the third if-else if statement along all
     the other formats. 

8. ``test/test_disk_io.py``

   * Add a reader for your packet format. This reader will be what is used in the actual
     code as well. It contains the sequence callback that we declared in the ``src/bifrost/packet_capture.h``
     file. Note that the header in this sequence callback is the ring header not the
     packet header. 
   * You will also need to add a ``_get_mypacket_data``, ``test_write_mypacket``,
     and ``test_read_mypacket``. 

9. ``test/test_udp_io.py``

   * The UDP version of ``test_disk_io``. Once you have written the disk I/O test, this
     test is fairly simple to implement, provided you wrote it correctly.
