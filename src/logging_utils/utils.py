import os
import datetime

def remove_outdated_logs(log_folder):
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    file_names = os.listdir(log_folder)
    current_date = datetime.datetime.today()
    for fname in file_names:
        date_string = fname.split('.')[-2].split('_')[-1]
        date = datetime.datetime.strptime(date_string, "%Y%m%d")
        gap = current_date - date
        if gap > datetime.timedelta(days=30):
            os.remove(os.path.join(log_folder, fname))