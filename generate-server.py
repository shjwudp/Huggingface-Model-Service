# Copyright (c) 2022 Jianbin Chang

import traceback
import time
import argparse

from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, AutoTokenizer


class GPTGenerate(Resource):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def post(self):
        start_time = time.time()
        try:
            request_j = request.get_json()
            context = request_j["context"]
            del request_j["context"]
            if "output_logits" in request_j:
                output_logits = request_j["output_logits"]
                del request_j["output_logits"]
            else:
                output_logits = False

            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.model.device)
            request_j["inputs"] = input_ids

            result = self.model.generate(**request_j)

            if request_j.get("return_dict_in_generate", False):
                sequences = result["sequences"]
                sequences = self.tokenizer.batch_decode(
                    sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output = dict(
                    sequences=sequences,
                    sequences_scores=result["sequences_scores"].tolist(),
                )
                if output_logits:
                    output.update(dict(
                        scores=[x.tolist for x in result["scores"]],
                        beam_indices=result["beam_indices"].tolist(),
                    ))
            else:
                generate_ids = result
                output = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            resp = {
                "compute_time": time.time() - start_time,
                "context": context,
                "output": output,
            }
            return make_response(jsonify(resp), 200)
        except:
            traceback.print_exc()
            return traceback.format_exc(), 400


class GenerateServer(object):
    def __init__(self, model, tokenizer):
        self.app = Flask(__name__)
        api = Api(self.app)
        api.add_resource(GPTGenerate, '/generate',
                         resource_class_args=[model, tokenizer])

    def run(self, url, port):
        self.app.run(url, debug=False, port=port)


def load_model(huggingface_model, model_type):
    assert model_type in ["CLM", "T5", "EVA"]
    if model_type == "CLM":
        model = AutoModelForCausalLM.from_pretrained(huggingface_model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    elif model_type == "T5":
        model = T5ForConditionalGeneration.from_pretrained(huggingface_model).cuda()
        # TODO: T5 model supports automap
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    elif model_type == "EVA":
        import sys
        sys.path.insert(0, "./third-party/EVA/src")
        from model import EVAModel, EVATokenizer
        model = EVAModel.from_pretrained(huggingface_model).cuda()
        tokenizer = EVATokenizer.from_pretrained(huggingface_model)
        sys.path = sys.path[1:]

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser("Huggingface Generate Model Service")
    parser.add_argument("--huggingface_model", required=True)
    parser.add_argument("--port", default=55556, type=int)
    parser.add_argument("--model_type", choices=["CLM", "T5", "EVA"], default="CLM")
    args = parser.parse_args()

    model, tokenizer = load_model(args.huggingface_model, args.model_type)
    generate_server = GenerateServer(model, tokenizer)

    generate_server.run("0.0.0.0", args.port)


if __name__ == "__main__":
    main()
