"""Specialize the original 320px ONNX graph for smaller native center views.

Convolution weights and numeric precision are preserved. Only static spatial
shapes, detection anchors and their stride tensors change. ONNX is a build-time
dependency; neither this builder nor the runtime imports PyTorch/Ultralytics.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PRESETS={'tight':(256,192,.20),'fast':(192,128,.15)}


def build_model(source, destination, width, height, head_confidence):
    import onnx
    import numpy as np
    from onnx import numpy_helper
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination:
        raise ValueError('The candidate must have its own output file')
    model=onnx.load(source)
    shape=[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    output=[d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    if shape!=[1,3,320,320] or output!=[1,6,2100]:
        raise ValueError('This builder requires the original static320px Head/Body ONNX graph')
    count=sum((height//s)*(width//s) for s in (8,16,32))
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value=height
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value=width
    model.graph.output[0].type.tensor_type.shape.dim[2].dim_value=count
    del model.graph.value_info[:]
    grids=[];strides=[]
    for stride in (8,16,32):
        yy,xx=np.meshgrid(np.arange(height//stride)+.5,np.arange(width//stride)+.5,indexing='ij')
        grids.append(np.stack((xx,yy)).reshape(2,-1));strides.append(np.full(xx.size,stride))
    anchors=np.concatenate(grids,axis=1)[None].astype(np.float32)
    updated=set()
    for tensor in model.graph.initializer:
        array=numpy_helper.to_array(tensor).copy()
        if tensor.name.endswith('/attn/Constant_output_0'):
            array[-1]=(height//32)*(width//32);updated.add('attention_flat')
        elif tensor.name.endswith('/attn/Constant_3_output_0'):
            array[-2:]=[height//32,width//32];updated.add('attention_grid')
        elif array.dtype==np.int64 and np.any(array==2100):
            array[array==2100]=count;updated.add('reshape')
        elif array.shape==(1,2,2100):
            array=anchors;updated.add('anchors')
        elif array.shape==(1,2100):
            array=np.concatenate(strides)[None].astype(array.dtype);updated.add('strides')
        else:
            continue
        tensor.CopyFrom(numpy_helper.from_array(array,tensor.name))
    if updated!={'attention_flat','attention_grid','reshape','anchors','strides'}:
        raise ValueError(f'The source graph layout is unsupported: {sorted(updated)}')
    metadata={entry.key:entry.value for entry in model.metadata_props}
    metadata.update(imgsz=f'[{height}, {width}]',head_confidence=str(head_confidence),
                    capture_width=str(width*2),capture_height=str(height*2),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    onnx.helper.set_model_props(model,metadata)
    model=onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    temporary=destination.with_suffix('.tmp.onnx')
    try:
        onnx.save(model,temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).with_name('best.onnx'))
    args=parser.parse_args()
    for name,(width,height,threshold) in PRESETS.items():
        destination=args.source.with_name(f'best-focus-{width}x{height}.onnx')
        print(build_model(args.source,destination,width,height,threshold))


if __name__=='__main__':main()
