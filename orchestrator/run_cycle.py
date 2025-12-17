import os, json, time, subprocess, sys, asyncio
import yaml, requests
from openai import OpenAI

# Try to import MCP. If not installed, we'll warn.
try:
    from mcp import StdioServerParameters, ClientSession
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("!! 'mcp' package not found. Integrations will be disabled.")
    print("!! Run: pip install mcp")

ROOT = os.path.dirname(os.path.dirname(__file__))

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_yaml(path: str) -> dict:
    return yaml.safe_load(read_text(path))

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Sync Wrappers for File/Git ---

class GitRepo:
    def __init__(self, path: str, allowed_commands: list[str]):
        self.path = path
        self.allowed = allowed_commands

    def _run(self, cmd_str: str, check=True) -> subprocess.CompletedProcess:
        # Security check (basic)
        if not cmd_str.startswith("git "):
             if cmd_str not in self.allowed:
                 raise RuntimeError(f"Command not allowed: {cmd_str}")
        
        print(f"[$] {cmd_str}")
        return subprocess.run(cmd_str, cwd=self.path, shell=True, capture_output=True, text=True)

    def is_clean(self) -> bool:
        p = self._run("git status --porcelain")
        return len(p.stdout.strip()) == 0

    def get_current_branch(self) -> str:
        p = self._run("git branch --show-current")
        return p.stdout.strip()

    def checkout_new_branch(self, name: str):
        self._run(f"git checkout -b {name}")

    def checkout(self, name: str):
        self._run(f"git checkout {name}")

    def add_all(self):
        self._run("git add .")

    def commit(self, msg: str):
        safe_msg = msg.replace('"', '\\"')
        self._run(f'git commit -m "{safe_msg}"')

    def push(self, branch: str):
        self._run(f"git push -u origin {branch}")

    def apply_patch(self, patch_path: str) -> bool:
        abs_patch = os.path.abspath(patch_path)
        p = self._run(f'git apply "{abs_patch}"', check=False)
        if p.returncode != 0:
            print(f"Patch apply failed:\n{p.stderr}")
            return False
        return True

    def discard_changes(self):
        self._run("git checkout .")
        self._run("git clean -fd")

    def run_command_allowlisted(self, cmd: str) -> str:
        if cmd not in self.allowed:
            return f"BLOCKED: {cmd}"
        p = self._run(cmd, check=False)
        return f"$ {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\nRC={p.returncode}\n"

# --- Async LLM & Tool Logic ---

def simple_llm_call(client: OpenAI, model: str, system_prompt: str, user_content: str, temperature: float) -> str:
    print(f"  > Calling LLM ({model})... (PROMPT: {len(user_content)} chars)")
    start_t = time.time()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    elapsed = time.time() - start_t
    print(f"  > LLM finished in {elapsed:.1f}s")
    return resp.choices[0].message.content

async def run_mcp_operation(client: OpenAI, model: str, system_prompt: str, user_content: str, temp: float, mcp_config: dict):
    # This function handles the complex "Research" step with tools
    # 1. Setup MCP connections
    start_t = time.time()
    if not MCP_AVAILABLE:
        return simple_llm_call(client, model, system_prompt, user_content, temp)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # Connect to servers defined in config
    from contextlib import AsyncExitStack
    async with AsyncExitStack() as stack:
        active_tools = []
        sessions = []

        # Start all servers
        mcp_servers = mcp_config.get('mcp_servers', {})
        if not mcp_servers:
             # Fast path if config exists but is empty
             return simple_llm_call(client, model, system_prompt, user_content, temp)

        print(f"  > Initializing {len(mcp_servers)} MCP servers...")
        for name, cfg in mcp_servers.items():
            try:
                # Create params
                env = os.environ.copy()
                env.update(cfg.get('env', {}))
                
                # Check command existence roughly (cmd.exe might fail to find npx if not in path, but usually it works)
                params = StdioServerParameters(command=cfg['command'], args=cfg['args'], env=env)
                
                # Connect
                stdio_ctx = stdio_client(params)
                read, write = await stack.enter_async_context(stdio_ctx)
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                # List tools
                result = await session.list_tools()
                
                # Context Optimization: Filter tools to prevent overflow
                # GitHub exposes ~26 tools. We only need the read/search ones for Research.
                ALLOWED_GITHUB = ['search_issues', 'get_issue', 'list_issues', 'search_repositories', 'read_file']
                
                for t in result.tools:
                    # Filter: If it's github, only allow research tools
                    if name == 'github' and t.name not in ALLOWED_GITHUB:
                        continue
                        
                    # Convert to OpenAI format
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    }
                    active_tools.append(tool_def)
                    # Track which session owns which tool
                    # We monkeypatch for local usage
                    t.session = session 
                
                sessions.append({"name": name, "session": session, "tool_names": [t.name for t in result.tools if (name != 'github' or t.name in ALLOWED_GITHUB)]})
                print(f"    + Connected to {name} (Loaded {len([t for t in result.tools if (name != 'github' or t.name in ALLOWED_GITHUB)])} tools)")
            except Exception as e:
                print(f"!! Failed to connect to {name}: {e}")

        # If no tools, fall back to simple call
        if not active_tools:
            return simple_llm_call(client, model, system_prompt, user_content, temp)

        # Loop for tool use (max 5 turns)
        print(f"  > Calling LLM ({model}) with {len(active_tools)} tools...")
        
        for turn in range(5):
            # OPENAI CALL
            resp = client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=messages,
                tools=active_tools
            )
            msg = resp.choices[0].message
            messages.append(msg) # Add assistant message to history

            if msg.tool_calls:
                print(f"    ! LLM requested {len(msg.tool_calls)} tools:")
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"      - {fn_name}({str(args)[:50]}...)")
                    
                    # Find session
                    target_session = None
                    for s in sessions:
                        if fn_name in s['tool_names']:
                            target_session = s['session']
                            break
                    
                    if target_session:
                        try:
                            # Execute
                            mcp_res = await target_session.call_tool(fn_name, arguments=args)
                            # Result is a list of content (Text/Image)
                            content_str = ""
                            for c in mcp_res.content:
                                if c.type == 'text': content_str += c.text
                                elif c.type == 'image': content_str += "[Image]"
                                elif c.type == 'resource': content_str += "[Resource]"
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content_str
                            })
                        except Exception as e:
                            print(f"      !! Tool Error: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Error: {e}"
                            })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Error: Tool not found in active sessions"
                        })
                # Loop continues to send tool outputs back to LLM
            else:
                elapsed = time.time() - start_t
                print(f"  > LLM finished in {elapsed:.1f}s")
                return msg.content

    return "Error: MCP Loop finished without final content."

def ensure_dirs():
    os.makedirs(os.path.join(ROOT, 'artifacts'), exist_ok=True)

def clean_json(text):
    if "<think>" in text and "</think>" in text: text = text.split("</think>")[-1]
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1: return text[start:end+1]
    return text.strip()

async def async_main():
    ensure_dirs()
    print("==> Starting Agent Cycle (v2 - MCP Enabled)...")
    
    models_cfg = load_yaml(os.path.join(ROOT, 'models.yaml'))
    target_cfg = load_yaml(os.path.join(ROOT, 'targets', 'mindbridge.yaml'))
    
    # Load Integrations
    int_cfg_path = os.path.join(ROOT, 'integrations_config.json')
    integrations = {}
    if os.path.exists(int_cfg_path):
        try:
            integrations = load_json(int_cfg_path)
            print("    -> Loaded integrations_config.json")
        except:
            print("!! Failed to parse integrations_config.json")
    
    base_url = models_cfg['lmstudio']['base_url']
    print(f"==> Connecting to LM Studio at {base_url}...")
    try:
        client = OpenAI(base_url=base_url, api_key='unused', timeout=300.0)
        client.models.list() 
        print("    -> Connected.")
    except Exception as e:
        print(f"!! Error connecting to LM Studio: {e}")
        return

    repo_path = target_cfg['repo_path']
    allowed = target_cfg['allowed_commands']
    repo = GitRepo(repo_path, allowed)

    if not repo.is_clean():
        print("!! Target repo is not clean. Commit or stash changes first.")
        # return # Strict mode

    ts = int(time.time())
    branch_name = f"agent-run-{ts}"
    print(f"==> Creating branch: {branch_name}")
    repo.checkout_new_branch(branch_name)

    # Load Roles
    manager_prompt = read_text(os.path.join(ROOT, 'roles', 'manager.md'))
    dev_prompt = read_text(os.path.join(ROOT, 'roles', 'developer.md'))
    tester_prompt = read_text(os.path.join(ROOT, 'roles', 'tester.md'))
    researcher_prompt = read_text(os.path.join(ROOT, 'roles', 'researcher.md'))

    # 1. Research (NOW WITH TOOLS via MCP)
    print("==> Roles: Researcher (MCP Enabled)")
    # We pass the MCP config here
    research = await run_mcp_operation(
        client,
        models_cfg['roles']['researcher']['model'],
        researcher_prompt,
        'Use your tools to find 3 high-impact improvements. Prefer GitHub issues or Competitor Gaps.',
        models_cfg['roles']['researcher']['temperature'],
        integrations
    )

    # 2. Base Checks
    print("==> Running base checks...")
    logs = ''
    for cmd in [target_cfg['install_cmd'], target_cfg['lint_cmd'], target_cfg['typecheck_cmd'], target_cfg['test_cmd']]:
        logs += repo.run_command_allowlisted(cmd) + '\n'

    # 3. Tester
    print("==> Roles: Tester")
    tester_report = simple_llm_call(
        client,
        models_cfg['roles']['tester']['model'],
        tester_prompt,
        f"Research:\n\n{research}\n\nLogs:\n\n{logs}",
        models_cfg['roles']['tester']['temperature']
    )

    # 4. Manager
    print("==> Roles: Manager")
    manager_out = simple_llm_call(
        client,
        models_cfg['roles']['manager']['model'],
        manager_prompt,
        f"Research:\n\n{research}\n\nTest Report:\n\n{tester_report}\n\nPick next tasks.",
        models_cfg['roles']['manager']['temperature']
    )
    
    try:
        plan = json.loads(clean_json(manager_out))
    except Exception as e:
        print(f"!! Manager JSON Error: {e}")
        plan = {"tasks": []}

    # 5. Developer
    print("==> Roles: Developer")
    dev_content = f"Tasks JSON:\n{json.dumps(plan, indent=2)}\n\nGenerate unified diff patch."
    patch = simple_llm_call(
        client,
        models_cfg['roles']['developer']['model'],
        dev_prompt,
        dev_content,
        models_cfg['roles']['developer']['temperature']
    )

    # Save
    patch_path = os.path.join(ROOT, 'artifacts', f'patch_{ts}.diff')
    with open(os.path.join(ROOT, 'artifacts', f'research_{ts}.md'), 'w', encoding='utf-8') as f: f.write(str(research))
    with open(os.path.join(ROOT, 'artifacts', f'test_report_{ts}.md'), 'w', encoding='utf-8') as f: f.write(tester_report)
    with open(os.path.join(ROOT, 'artifacts', f'plan_{ts}.json'), 'w', encoding='utf-8') as f: json.dump(plan, f, indent=2)
    with open(patch_path, 'w', encoding='utf-8') as f: f.write(patch)
    print(f"Artifacts saved to artifacts/")

    # 6. Apply
    print("==> Applying Patch...")
    if not repo.apply_patch(patch_path):
        print("!! Patch application failed.")
        return

    # 7. Verify
    print("==> Verifying changes...")
    verify_log = repo.run_command_allowlisted(target_cfg['test_cmd'])
    verification_passed = "RC=0" in verify_log and "FAIL" not in verify_log 
    
    if verification_passed:
        print("SUCCESS: Verification passed.")
        repo.add_all()
        tasks_doc = ", ".join([t.get('id', 'Task') for t in plan.get('tasks', [])])
        repo.commit(f"Agent-Factory: Implemented {tasks_doc}")
        repo.push(branch_name)
        print(f"\nDraft PR Ready! Branch: {branch_name}")
        print(f"https://github.com/Romone6/MindBridge/compare/main...{branch_name}?expand=1\n")
    else:
        print("FAILURE: Verification failed. Reverting...")
        repo.discard_changes()

def main():
    asyncio.run(async_main())

if __name__ == '__main__':
    main()
