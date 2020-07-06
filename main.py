from correlation_modeling_main import run_correlation_modeling_main
from File_Management.path_and_file_management_Func import remove_win10_max_path_limit
import time
from datetime import datetime
import numpy as np

"""
This (remove_win10_max_path_limit) is a default call to make the code run on Windows platform 
(and only Windows 10 is supported!)
Because the Windows 10 default path length limitation (MAX_PATH) is 256 characters, many load/save functions in
this project may have errors in reading the path
To restore the path_limit, call File_Management.path_management_Func.restore_win10_max_path_limit yourself
"""
remove_win10_max_path_limit()


# %% Set different random seed according to datetime.now()
np.random.seed(int(time.mktime(datetime.now().timetuple())))

# %% Run correlation modeling
run_correlation_modeling_main()
