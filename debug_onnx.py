from statistics import mode
import torch 
import onnx 
import onnxruntime 
import numpy as np 
from types import MethodType
from application.model import*
from application.preprocess import*
 
class DebugOp(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, x, name): 
        return x 
 
    @staticmethod 
    def symbolic(g, x, name): 
        return g.op("my::Debug", x, name_s=name) 
 
debug_apply = DebugOp.apply 
 
class Debugger(): 
    def __init__(self): 
        super().__init__() 
        self.torch_value = dict() 
        self.onnx_value = dict() 
        self.output_debug_name = [] 
 
    def debug(self, x, name): 
        self.torch_value[name] = x.detach().cpu().numpy() 
        return debug_apply(x, name) 
 
    def extract_debug_model(self, input_path, output_path): 
        model = onnx.load(input_path) 
        inputs = [input.name for input in model.graph.input] 
        outputs = [] 
 
        for node in model.graph.node: 
            if node.op_type == 'Debug': 
                debug_name = node.attribute[0].s.decode('ASCII') 
                self.output_debug_name.append(debug_name) 
 
                output_name = node.output[0] 
                outputs.append(output_name) 
 
                node.op_type = 'Identity' 
                node.domain = '' 
                del node.attribute[:] 
        e = onnx.utils.Extractor(model) 
        extracted = e.extract_model(inputs, outputs) 
        onnx.save(extracted, output_path) 
 
    def run_debug_model(self, input, debug_model): 
        sess = onnxruntime.InferenceSession(debug_model,  
        providers=['CPUExecutionProvider']) 
 
        onnx_outputs = sess.run(None, input) 
        for name, value in zip(self.output_debug_name, onnx_outputs): 
            self.onnx_value[name] = value 
 
    def print_debug_result(self): 
        for name in self.torch_value.keys(): 
            if name in self.onnx_value: 
                mse = np.mean(self.torch_value[name] - self.onnx_value[name])**2
                print(f"{name} MSE: {mse}")


if __name__ == '__main__':
    debugger = Debugger()

    num_examples=100
    source,target=preprocess(num_examples)

    enc_inputs, dec_inputs, dec_outputs, src_vocab_size, tgt_vocab_size, src_vocab, tgt_vocab = make_data(20, source, target)

    src_w2i = src_vocab.token_to_idx
    src_i2w = {i: w for i, w in enumerate(src_vocab.idx_to_token)}

    w2i = tgt_vocab.token_to_idx
    i2w = {i: w for i, w in enumerate(tgt_vocab.idx_to_token)}

    # 加载预训练模型
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
    # model.load_state_dict(torch.load("application/saved/model-withMasked.pt"))

    def new_forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_outputs = debugger.debug(enc_outputs, 'enc_outputs')
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_outputs = debugger.debug(dec_outputs, 'dec_outputs')
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = debugger.debug(dec_logits, 'dec_logits')
        return dec_logits.view(-1, dec_logits.size(-1))

    model.forward = MethodType(new_forward, model)

    dummy_enc_input = torch.randint(0, 10, (1, 20))
    dummy_dec_input = torch.randint(0, 10, (1, 5))

    torch.onnx.export(model, (dummy_enc_input, dummy_dec_input), 'before_debug.onnx', opset_version=11, do_constant_folding=True, input_names=['enc_input','dec_input']) 
    debugger.extract_debug_model('before_debug.onnx', 'after_debug.onnx') 
    debugger.run_debug_model({'enc_input':dummy_enc_input.numpy(), 'dec_input':dummy_dec_input.numpy()}, 'after_debug.onnx') 
    debugger.print_debug_result() 

