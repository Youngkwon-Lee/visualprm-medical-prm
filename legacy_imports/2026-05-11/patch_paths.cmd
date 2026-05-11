@echo off
py -c "from pathlib import Path; p=Path(r'D:\visualprm\experiments\rag_gym_visualprm_loop\run_prm_compare_pathvqa4.py'); t=p.read_text(encoding='utf-8'); t=t.replace('/mnt/d/visualprm','D:/visualprm'); p.write_text(t,encoding='utf-8'); print('patched')"
