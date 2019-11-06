import pyopencl as cl
import numpy as np
import time
import os 

os.environ['PYOPENCL_CTX'] = '0' # NVIDIA = 0, INTEL = 1

def passage(xxx):
	A = np.random.rand(xxx).astype(np.float32)
	B = np.random.rand(xxx).astype(np.float32)
	C = np.empty_like(A)

	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)

	#ctx = cl.Context()
	#queue = cl.CommandQueue(ctx)

	A_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
	B_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
	C_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, A.nbytes)
	prg = cl.Program(ctx, """
		__kernel
		void sum(__global const float *a, __global const float *b, __global float *c){
			int i = get_global_id(0);
			for (int x = 0; x< 100; x++){
				c[i] = a[i] / b[i];
			}
		}
	""").build()
	prg.sum(queue, A.shape, None, A_buf, B_buf, C_buf)
	cl.enqueue_read_buffer(queue, C_buf, C).wait()
	
def passage_cpu(xxx):
	A = np.random.rand(xxx).astype(np.float32)
	B = np.random.rand(xxx).astype(np.float32)
	C = np.empty_like(A)
	for x in range(0,100):
		C = A / B
y=1
for x in range(1,12):
	y = x*y
	t1 = time.time()
	passage(y)
	print y,":",(time.time()-t1),"\n"

y=1
for x in range(1,12):
	y = x*y
	t1 = time.time()
	passage_cpu(y)
	print y,":",(time.time()-t1),"\n"

	
