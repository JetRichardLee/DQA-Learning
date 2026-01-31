import os
import torch
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics,attack_by_DQA,attack_by_GCG,prada_finetune,attack_by_poison,prada_finetune_s,attack_by_DQA_h,prada_finetune_jina,prada_finetune_ibm,attack_by_S_GCG
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers.cache_utils import DynamicCache
import random
DynamicCache.get_usable_length = DynamicCache.get_seq_length


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl','jinaai','qwen3I','qwen2I','ibm',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge','gemma','qwen3','nvidia'])
                                 
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    score_file_path = os.path.join(args.output_dir,f'score.json')

    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None:
        examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
    else:
        examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents',cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
    doc_ids = []
    documents = []
    mapping_doc = {}
    for dp in doc_pairs:
        #print(dp['id'])
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
        mapping_doc[dp['id']]=len(doc_ids)-1

    surrogate_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="auto"
        )
    surrogate_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

    """
    # If parada, use here and later apply GCG
    sample_queries = [] 
    cnt=0
    for e in examples:
        cnt+=1
        sample_queries.append(e["query"])

    sample_queries= random.sample(sample_queries, k=int(len(sample_queries)*0.1))
    sample_documents= random.sample(documents, k=int(len(documents)*0.1))
    
    with open(os.path.join(args.config_dir,args.model,f"{args.task}.json")) as f:
        config = json.load(f)
    if args.model=='sf':
        target_tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        target_model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        surrogate_model =prada_finetune(surrogate_model,target_model,surrogate_tokenizer,target_tokenizer,sample_queries,sample_documents,num_epochs=3, batch_size=1)    
    elif args.model=='qwen':
        target_tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        target_model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        surrogate_model =prada_finetune(surrogate_model,target_model,surrogate_tokenizer,target_tokenizer,sample_queries,sample_documents,num_epochs=3, batch_size=1)   
    elif args.model=='e5':
        target_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        target_model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        surrogate_model =prada_finetune(surrogate_model,target_model,surrogate_tokenizer,target_tokenizer,sample_queries,sample_documents,num_epochs=3, batch_size=1)    
    elif args.model=='gemma':
        target_model = SentenceTransformer('google/embeddinggemma-300m')
        instructions=config['instructions_long'] if args.long_context else config['instructions']
        surrogate_model =prada_finetune_s(surrogate_model,target_model,surrogate_tokenizer,sample_queries,sample_documents,instructions,num_epochs=3, batch_size=1)    
    elif args.model=='qwen3':
        target_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        instructions=config['instructions_long'] if args.long_context else config['instructions']
        surrogate_model =prada_finetune_s(surrogate_model,target_model,surrogate_tokenizer,sample_queries,sample_documents,instructions,num_epochs=3, batch_size=1)    
    elif args.model=='jinaai':
        target_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3',device_map="auto", trust_remote_code=True)
        instructions=config['instructions_long'] if args.long_context else config['instructions']
        surrogate_model =prada_finetune_jina(surrogate_model,target_model,surrogate_tokenizer,sample_queries,sample_documents,instructions,num_epochs=3, batch_size=1)    
    elif args.model=='ibm':
        target_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
        instructions=config['instructions_long'] if args.long_context else config['instructions']
        surrogate_model =prada_finetune_ibm(surrogate_model,target_model,surrogate_tokenizer,sample_queries,sample_documents,instructions,num_epochs=3, batch_size=1)    

    del target_model
    torch.cuda.empty_cache()
    #"""
    if True:#not os.path.isfile(score_file_path):
        with open(os.path.join(args.config_dir,args.model,f"{args.task}.json")) as f:
            config = json.load(f)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
        if args.long_context:
            key = 'gold_ids_long'
        else:
            key = 'gold_ids'
        
        cnt=0
        for e in tqdm(examples):
        
            queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            #"""
            for doc_id in e[key]:
                index = mapping_doc[doc_id]

                documents[index],_ = attack_by_GCG(documents[index],e["query"],surrogate_model,surrogate_tokenizer,epochs=20,num_q=1)
                #documents[index],_ = attack_by_DQA(documents[index],surrogate_model,surrogate_tokenizer,num_sts_tokens=20,num_q=10,epochs=20,q_rate=0.3,maxs=True)
                #documents[index],_ = attack_by_DQA_h(documents[index],surrogate_model,surrogate_tokenizer,num_q=10,epochs=20,q_rate=0.3,maxs=True)
                #documents[index],_ = attack_by_S_GCG(documents[index],surrogate_model,surrogate_tokenizer,num_q=10,epochs=20,q_rate=0.3)
                #documents[index],_ = attack_by_poison(documents[index],e["query"],surrogate_model,surrogate_tokenizer,epochs=20,num_q=1)


                torch.cuda.empty_cache()

            assert len(overlap)==0
        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"

        del surrogate_model

        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")):
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")) as f:
                cached_doc_ids = json.load(f)
            for id1,id2 in zip(cached_doc_ids,doc_ids):
                assert id1==id2
        else:
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json"),'w') as f:
                json.dump(doc_ids,f,indent=2)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        

        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})

        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries, query_ids=query_ids, documents=documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )

        #print(scores)
        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')
    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
