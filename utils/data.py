import os


def video_paths(root):
    paths = []
    excluded_tasks = ['head', 'head2', 'head3']
    subjects = sorted(os.listdir(root))
    for sub in subjects:
        sub_dir = f"{root}/{sub}/video"
        for task in sorted(os.listdir(sub_dir)):
            paths.append(f"{sub_dir}/{task}") if task not in excluded_tasks \
            else None
    return paths