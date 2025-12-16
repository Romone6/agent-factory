import os, json, time, subprocess
import yaml, requests
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(__file__))

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_yaml(path: str) -> dict:
    return yaml.safe_load(read_text(path))

def blocked_path(path: str, deny_prefixes: list[str]) -> bool:
    p = path.replace('\\', '/').lower()
    for d in deny_prefixes:
        if p.startswith(d.lower()):
            return True
    return False

def run_allowed(cmd: str, cwd: str, allowed: list[str]) -> str:
    if cmd not in allowed:
        raise RuntimeError(f'Blocked command (not allowlisted): {cmd}')
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    return f"$ {cmd}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n\nRC={p.returncode}\n"

def llm_call(client: OpenAI, model: str, system_prompt: str, user_content: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content

def ensure_dirs():
    os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)

def main():
    ensure_dirs()
    models_cfg = load_yaml(os.path.join(ROOT, 'models.yaml'))
    target_cfg = load_yaml(os.path.join(ROOT, 'targets', 'mindbridge.yaml'))

    base_url = models_cfg['lmstudio']['base_url']
    client = OpenAI(base_url=base_url, api_key='unused')

    repo_path = target_cfg['repo_path']
    allowed = target_cfg['allowed_commands']

    # Pull role prompts
    manager_prompt = read_text(os.path.join(ROOT, 'roles', 'manager.md'))
    dev_prompt = read_text(os.path.join(ROOT, 'roles', 'developer.md'))
    tester_prompt = read_text(os.path.join(ROOT, 'roles', 'tester.md'))
    researcher_prompt = read_text(os.path.join(ROOT, 'roles', 'researcher.md'))

    # 1) Research brief (v0: no live browsing here; you can paste URLs in later)
    research = llm_call(
        client,
        models_cfg['roles']['researcher']['model'],
        researcher_prompt,
        'Generate an actionable research brief for improvements/features. Output markdown only.',
        models_cfg['roles']['researcher']['temperature']
    )

    # 2) Run checks
    logs = ''
    for cmd in [target_cfg['install_cmd'], target_cfg['lint_cmd'], target_cfg['typecheck_cmd'], target_cfg['test_cmd']]:
        logs += run_allowed(cmd, repo_path, allowed) + '\n'

    # 3) Tester report
    tester_report = llm_call(
        client,
        models_cfg['roles']['tester']['model'],
        tester_prompt,
        f"Research:\n\n{research}\n\nLogs:\n\n{logs}",
        models_cfg['roles']['tester']['temperature']
    )

    # 4) Manager tasks
    manager_out = llm_call(
        client,
        models_cfg['roles']['manager']['model'],
        manager_prompt,
        f"Research:\n\n{research}\n\nTest Report:\n\n{tester_report}\n\nPick next tasks.",
        models_cfg['roles']['manager']['temperature']
    )
    plan = json.loads(manager_out)

    # 5) Developer patch
    patch = llm_call(
        client,
        models_cfg['roles']['developer']['model'],
        dev_prompt,
        f"Tasks JSON:\n{json.dumps(plan, indent=2)}\n\nGenerate unified diff patch.",
        models_cfg['roles']['developer']['temperature']
    )

    ts = int(time.time())
    with open(os.path.join(ROOT, 'artifacts', f'research_{ts}.md'), 'w', encoding='utf-8') as f:
        f.write(research)
    with open(os.path.join(ROOT, 'artifacts', f'test_report_{ts}.md'), 'w', encoding='utf-8') as f:
        f.write(tester_report)
    with open(os.path.join(ROOT, 'artifacts', f'plan_{ts}.json'), 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(ROOT, 'artifacts', f'patch_{ts}.diff'), 'w', encoding='utf-8') as f:
        f.write(patch)

    print('Wrote artifacts to /artifacts')
    print('Next step (manual): inspect patch then apply in target repo with: git apply <patchfile>')

if __name__ == '__main__':
    main()
