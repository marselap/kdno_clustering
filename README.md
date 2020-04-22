KDNO clustering and optimization python scripts


Use:
```
chmod +x scripts/usage.py

python scripts/usage.py [input_file/s/folder] [output_folder] varname=[unused]
```

Input can be a file, several files (delimited by space), or a folder. 
The input file can be a .mat with "CalTx" matrix of transforms, or .mat with "data" of [pos, quat] rows. 
If input is not in mx7 (pos,quat) form, only CalTx is used. 

TODO master, slave, TP_rec.

Output is path to folder if lines (see below) are uncommented. Otherwise unused
    #     clustering.save_plot()
    #     clustering.save_to_mat()

save_to_mat will save a mat file under *input_name*_[n_clusters].mat containing c_c matrix of cluster centres. 
Size is (4,4,n_clusters)

varname is unused, intended for TODO with master, slave tp_rec

```
python scripts/usage.py ./data/raw_T output/ varname="CalTx"
python scripts/usage.py ./data/raw_T/VU_0406_EXP1_krug_vrtnja.mat output/ varname="dmy"
python scripts/usage.py ./data/raw_q/VU_0406_EXP1_krug_vrtnja.mat output/ varname="dmy"
```
