[Training Transformer models using Pipeline Parallelism — PyTorch Tutorials 2.1.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html)

```
docker run -it --name=cyy --privileged -p 15001:22 --gpus all -v /data/cyy/pipe:/share nvcr.io/nvidia/pytorch:23.09-py3 
docker exec -it --privileged cyy bash
passwd
#chmod o+w /tmp 
apt update
apt install openssh-server -y
vim /etc/ssh/sshd_config
PermitRootLogin yes
service ssh start

pip install 'portalocker>=2.0.0'
```

# checkpoint="never"

```
model = Pipe(torch.nn.Sequential(*module_list), chunks = chunks, checkpoint="never")
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/pipe.py

```
for device in self.devices:
	self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])
#[2,8]每个stage的每个micro的计算使用一个stream，用来拷贝上一个stage的结果

self.pipeline = Pipeline(self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop)
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/pipeline.py

```
(self.in_queues, self.out_queues) = create_workers(devices)
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/worker.py

```
from queue import Queue
from threading import Thread
"""Spawns worker threads. A worker thread is bound to a device."""
t = Thread(target=worker, args=(in_queue, out_queue, device), daemon=True,)
t.start()
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/pipe.py

```
        batches = microbatch.scatter(*inputs, chunks=self.chunks)

        # Run pipeline parallelism.
        self.pipeline.run(batches)

        # Merge the micro-batches into one mini-batch.
        output = microbatch.gather(batches)
        return RRef(output)
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/pipeline.py

self.pipeline.run:

```
        for schedule in _clock_cycles(m, n):
            self.fence(batches, schedule, skip_trackers)
            self.compute(batches, schedule, skip_trackers)
```

_clock_cycles:

```
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1)
    # 3 (3,0) (2,1)
    # 4            
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]
```

self.fence:

```
        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0 and j != 0:
                _depend(batches[i - 1], batches[i])#实现不清楚，为什么不包括j=0

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                _copy(batches[i], prev_stream, next_stream)
```

_depend(batches[i - 1], batches[i]):

(1,1)

batch[0] (fork_from),batch[1]

```
    fork_from_idx = fork_from.find_tensor_idx()#0
    join_to_idx = join_to.find_tensor_idx()#0

    fork_from[fork_from_idx], phony = fork(fork_from[fork_from_idx])
    join_to[join_to_idx] = join(join_to[join_to_idx], phony)
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/dependency.py

fork:

(0,1)

```
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
```

Fork.forward:

```
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/phony.py

get_phony:

(device0,0)->tensor0

(device0,1)->tensor1

(device1,0)->tensor2

(device1,1)->tensor3

```
    key = (device, requires_grad)

    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(0, device=device, requires_grad=requires_grad)

        _phonies[key] = phony
```

Fork.backward

```
return grad_input
```

join:

```
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)
```

Join:

```
return input.detach()
```

```
return grad_input, None
```

_copy(batches[i], prev_stream, next_stream):

```
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in batch])
```

Copy.forward:

```
        output_stream = current_stream(get_device(next_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                if torch.is_tensor(x):
                    y = x.to(get_device(next_stream), non_blocking=True)
                    output.append(y)

                    # 'prev_stream' is not where 'x' has been allocated.
                    record_stream(x, prev_stream)
                    # 'y' has been allocated on 'next_stream'.
                    # It might be used on the current stream captured as 'output_stream'.
                    record_stream(y, output_stream)
                else:
                    output.append(x)

        return tuple(output)
```

Tensor.record_stream(*stream*)

Ensures that the tensor memory is not reused for another tensor until all current work queued on `stream` are complete.

Copy.backward:

```
        grad_input: Deque[Tensor] = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x.to(get_device(prev_stream), non_blocking=True)
                grad_input.appendleft(y)

                # 'next_stream' is not where 'x' has been allocated.
                record_stream(x, next_stream)
                # 'y' has been allocated on 'prev_stream'.
                # It might be used on the current stream captured as 'input_stream'.
                record_stream(y, input_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
```

self.compute:

```
        streams = [current_stream(d) for d in devices]
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j])#null等copy接收数据
            
            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop
            if checkpoint:
            
            else:
                def compute(
                    batch: Batch = batch,
                    partition: nn.Module = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> Batch:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return batch.call(partition)

                task = Task(streams[j], compute=compute, finalize=None)

            # Compute tasks in parallel. ([2] in the diagram)
            self.in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = self.out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:
                _wait(batch, streams[j], copy_streams[j][i])#copy为什么要等null，null不是已经工作完了？难道null与cpu是异步？

            batches[i] = batch
```

worker:

处理task的线程的函数

一个stage由一个cpu线程负责提交作业到gpu，每个任务就是计算一个stage的一个micro

    with use_device(device):#使用设备
        while True:
            task = in_queue.get()
    
            if task is None:
                break
    
            try:
                batch = task.compute()
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue
    
            out_queue.put((True, (task, batch)))
    
    done = (False, None)
    out_queue.put(done)

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/microbatch.py

call:

```
return Batch(function(self._values))#grad_fn=NativeLayerNormBackward0
```

_wait(batch, streams[j], copy_streams[j][i]):

```
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)#grad_fn=WaitBackward
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in batch])
```

Wait(torch.autograd.Function):

forward:

forward有一个输入的requires_grad=True，输出就有grad_fn，是当前函数，next_functions是input.grad_fn

x.detach()创建一个新的tensor，共享存储，去掉grad_fn，使requires_grad=false，在里面detach有什么用？？？？？

```
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        wait_stream(next_stream, prev_stream)

        return tuple(x.detach() if torch.is_tensor(x) else x for x in input)
```

wait_stream(next_stream, prev_stream):

It makes the source stream(next_stream) wait until the target stream(prev_stream) completes work queued

在prev_stream完成当前工作前，next_stream不能再继续增加工作了，先前的工作可以继续进行

```
as_cuda(source).wait_stream(as_cuda(target))
```

backward:

```
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        wait_stream(prev_stream, next_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + grad_input#元组拼接
```

output = microbatch.gather(batches)

return RRef(output)

    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1)
    # 3 	  (2,1)
    # 4      
    output = torch.cat(tensors)

# checkpoint = "except_last"

compute:

```
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j])

            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop#这里有bug，checkpoint_stop=7=8-1，实际切分只有7个micro，永远成立，如果batch整除8就不会有这个问题
            if checkpoint:

                def function(
                    *inputs,
                    partition: nn.Module = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return partition(*inputs)

                chk = Checkpointing(function, batch)  # type: ignore[arg-type]
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:

            # Compute tasks in parallel. ([2] in the diagram)
            self.in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = self.out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:
                _wait(batch, streams[j], copy_streams[j][i])#grad_fn=wait

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            batches[i] = batch
```

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/worker.py

batch = task.compute():

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/checkpoint.py

compute=chk.checkpoint:

```
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)

        # Use a phony which requires grad to ensure that Checkpoint can be
        # tracked by the autograd engine even when none of the input tensors
        # require grad.
        phony = get_phony(self.batch.get_device(), requires_grad=True)

        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)#grad_fn=checkpoint

        # Gradients are only supported for float Tensors.
        if isinstance(output, tuple):
            output = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in output])

        return Batch(output)
```

Checkpoint.forward:

```
        with torch.no_grad(), enable_checkpointing():
            if input_atomic:
                assert len(inputs) == 1
                output = function(inputs[0])
            else:
                output = function(*inputs)
        return output
```

backward:

```
        output, input_leaf = ctx.recomputed.pop()

        if isinstance(output, tuple):
            outputs = output
        else:
            outputs = (output,)
        if any(torch.is_tensor(y) and y.requires_grad for y in outputs):
            tensors = tuple([x for x in outputs if torch.is_tensor(x) and x.requires_grad])
            torch.autograd.backward(tensors, grad_output)#后向

        grad_input: List[Optional[Tensor]] = [None, None, None, None, None]
        grad_input.extend(x.grad if torch.is_tensor(x) else None for x in input_leaf)
        return tuple(grad_input)
```

finalize=chk.recompute:

```
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)

        # Use a tensor in the batch to tie together fork-join
        tensor_idx = batch.find_tensor_idx()
        # batch[tensor_idx] is always requiring grad, because it has been passed
        # checkpoint with a phony requiring grad.
        batch[tensor_idx], phony = fork(batch[tensor_idx])#grad_fn=fork
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)
        batch[tensor_idx] = join(batch[tensor_idx], phony)
```

Recompute.forward:

```
        ctx.save_for_backward(*tensors)

        return phony
```

backward:

```
        with restore_rng_states(device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                if ctx.input_atomic:
                    assert len(inputs_leaf) == 1
                    output = ctx.function(inputs_leaf[0])#model.forward
                else:
                    output = ctx.function(*inputs_leaf)

        ctx.recomputed.append((output, inputs_leaf))
```

前向checkpoint是模型前向，后向recompute是模型后向之前的前向，后向checkpoint是模型后向

为什么要fork，直接recompute不行吗？

recompute和梯度跨stage传递重叠？

rpc没有什么用，注释掉没有任何影响

# 未看内容

/usr/local/lib/python3.10/dist-packages/torch/distributed/pipeline/sync/pipe.py

```
        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)
        self._skip_layout = inspect_skip_layout(self.partitions)
```

