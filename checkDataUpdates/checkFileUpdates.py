import difflib
import os

class SyncData:
    def __init__(self, folder,temp_folder):
        self.folder = folder
        self.temp_folder = temp_folder
        

    def _compareFiles(self):
        # Read the files
        with open(self.file1, 'r') as file1:
            file1_lines = file1.readlines()
        with open(self.file2, 'r') as file2:
            file2_lines = file2.readlines()

        differ = difflib.Differ()

        diff = list(differ.compare(file1_lines, file2_lines))
        diff_only = [line for line in diff if not line.startswith('  ')]

        return diff_only
    
    def compareFolders(self):
        # Compare the files in the folders
        diff=[]
        for file in os.listdir(self.folder):
            self.file1 = os.path.join(self.folder, file)
            self.file2 = os.path.join(self.temp_folder, file)
            diff.extend(self._compareFiles())
        return diff

    def syncTempFolder(self):
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        for file in os.listdir(self.folder):
            self.file1 = os.path.join(self.folder, file)
            self.file2 = os.path.join(self.temp_folder, file)
            with open(self.file1, 'r') as file1:
                file1_lines = file1.readlines()
            with open(self.file2, 'w') as file2:
                file2.writelines(file1_lines)
