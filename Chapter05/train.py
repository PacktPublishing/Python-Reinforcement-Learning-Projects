'''
Created on 31 May 2017

@author: ywz
'''
import argparse, os, sys, cluster
from six.moves import shlex_quote  #@UnresolvedImport

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num_workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-e', '--env', type=str, default="demo",
                    help="Environment")
parser.add_argument('-l', '--log_dir', type=str, default="save",
                    help="Log directory path")

def new_cmd(session, name, cmd, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))

def create_commands(session, num_workers, logdir, env, shell='bash'):

    base_cmd = ['CUDA_VISIBLE_DEVICES=',
                sys.executable, 
                'worker.py', 
                '--log_dir', logdir,
                '--num_workers', str(num_workers),
                '--env', env]

    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job_name", "ps"], logdir, shell)]
    for i in range(num_workers):
        cmd = base_cmd + ["--job_name", "worker", "--task", str(i)]
        cmds_map.append(new_cmd(session, "w-%d" % i, cmd, logdir, shell))
    cmds_map.append(new_cmd(session, "htop", ["htop"], logdir, shell))
    
    windows = [v[0] for v in cmds_map]
    notes = ["Use `tmux attach -t {}` to watch process output".format(session),
             "Use `tmux kill-session -t {}` to kill the job".format(session),
             "Use `ssh -L PORT:SERVER_IP:SERVER_PORT username@server_ip` to remote Tensorboard"]

    cmds = ["kill $(lsof -i:{}-{} -t) > /dev/null 2>&1".format(cluster.PORT, num_workers+cluster.PORT),
            "tmux kill-session -t {}".format(session),
            "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)]
    
    for w in windows[1:]:
        cmds.append("tmux new-window -t {} -n {} {}".format(session, w, shell))
    cmds.append("sleep 1")

    for _, cmd in cmds_map:
        cmds.append(cmd)
    return cmds, notes

def main():
    
    args = parser.parse_args()
    cmds, notes = create_commands("a3c", args.num_workers, args.log_dir, args.env)

    print("Executing the following commands:")
    print("\n".join(cmds))
    
    os.environ["TMUX"] = ""
    os.system("\n".join(cmds))
    
    print("Notes:")
    print('\n'.join(notes))
    
if __name__ == "__main__":
    main()
    
    