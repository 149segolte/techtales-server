import uuid
import gzip
import os
import sys
import shutil
import pickle
import git
import numpy as np

from sentence_transformers import SentenceTransformer, util
from textwrap import dedent
from fastapi import FastAPI, BackgroundTasks
from pathlib import Path
from tree_sitter import Tree
from tree_sitter_languages import get_parser
from tqdm import tqdm

app = FastAPI()
base_path = Path(__file__).parent / "data"
urls = {}
model_name = "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net"
model = SentenceTransformer(model_name)
nodes_to_extract = ['function_definition', 'method_definition', 'function_declaration', 'method_declaration']
batch_size = 32
n_results = 5


def _supported_file_extensions():
    return {
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.py': 'python',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.ktm': 'kotlin',
        '.php': 'php',
    }


def _traverse_tree(tree: Tree):
    cursor = tree.walk()
    reached_root = False
    while reached_root is False:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            if cursor.goto_next_sibling():
                retracing = False


def _extract_functions(nodes, fp, file_content, relevant_node_types):
    out = []
    for n in nodes:
        if n.type in relevant_node_types:
            node_text = dedent('\n'.join(file_content.split('\n')[
                               n.start_point[0]:n.end_point[0]+1]))
            out.append(
                {'file': fp, 'line': n.start_point[0], 'text': node_text})
    return out


def _get_repo_functions(root, supported_file_extensions, relevant_node_types):
    functions = []
    print('Extracting functions from {}'.format(root))
    for fp in tqdm([root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')]):
        if not os.path.isfile(fp):
            continue
        with open(fp, 'r') as f:
            lang = supported_file_extensions.get(fp[fp.rfind('.'):])
            if lang:
                parser = get_parser(lang)
                file_content = f.read()
                tree = parser.parse(bytes(file_content, 'utf8'))
                all_nodes = list(_traverse_tree(tree.root_node))
                functions.extend(_extract_functions(
                    all_nodes, fp, file_content, relevant_node_types))
    return functions


def _search(query_embedding, dataset, k=5):
    corpus_embeddings = dataset['embeddings']
    functions = dataset['functions']
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)), sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.append((score, functions[idx]))
    return out

def embed_repo(id, url):
    path = base_path / str(id) / "repo"
    path.mkdir(parents=True)
    if not os.path.isfile(path / '.embedding'):
        print('Embeddings not found in {}. Generating embeddings now.'.format(path))
        repo = git.Repo.clone_from(url, path)
        functions = _get_repo_functions(path, _supported_file_extensions(), nodes_to_extract)

        if not functions:
            print('No supported languages found in {}. Exiting'.format(path))
            return

        print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
            len(functions), int(np.ceil(len(functions)/batch_size))))
        corpus_embeddings = model.encode(
            [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

        dataset = {'functions': functions, 'embeddings': corpus_embeddings}
        with gzip.open(path + '/' + '.embedding', 'w') as f:
            f.write(pickle.dumps(dataset))
    return


@app.get("/")
def read_root():
    return {"Houston": "We are go for launch"}


@app.post("/new")
def add_repo(url: str, background_tasks: BackgroundTasks):
    user, repo = [s for s in url.split("/")[-2:] if s.replace("-", "").isalnum()]
    if not user or not repo:
        return {"error": "Invalid URL"}
    url = "https://github.com/" + user + "/" + repo
    id = urls.get(url)
    if id:
        return {"success": str(id)}
    id = uuid.uuid4()
    urls[url] = id
    new_dir = base_path / str(id)
    new_dir.mkdir(parents=True)
    with open(new_dir / "url.txt", "w") as f:
        f.write(url)
    background_tasks.add_task(embed_repo, id, url)
    return {"success": str(id)}


@app.get("/{repo_id}")
def read_status(repo_id: uuid.UUID):
    path = base_path / str(repo_id)
    if not path.exists():
        return {"error": "not found"}
    if not os.path.isfile(path / "repo" / '.embedding'):
        return {"status": "embedding"}
    return {"success": "processed"}


@app.delete("/{repo_id}")
def delete_repo(repo_id: uuid.UUID):
    path = base_path / str(repo_id)
    if not path.exists():
        return {"error": "not found"}
    shutil.rmtree(path)
    return {"success": "deleted"}


@app.get("/{repo_id}/query")
def query_repo(repo_id: uuid.UUID, query: str):
    path = base_path / str(repo_id)
    if not path.exists():
        return {"error": "not found"}
    if not query:
        return {"error": "empty query"}
    if not os.path.isfile(path / "repo" / '.embedding'):
        return {"error": "not processed"}
    with gzip.open(path / "repo" / '.embedding', 'r') as f:
        dataset = pickle.loads(f.read())
    qe = model.encode(query, convert_to_tensor=True)
    results = _search(qe, dataset, k=n_results)
    return results
