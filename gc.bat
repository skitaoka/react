@echo off

for /r /d %%d in (.svn;debug;release;ipch;x64) do (
  if exist "%%d" (
    rmdir /s /q "%%d"
    echo deleted: %%d
  )
)

for /r %%f in (*.suo;*.ncb;*.user;*.db;*.sdf;*.class;*.pdb;*.o;.DS_Store) do (
  if exist "%%f" (
    del "%%f"
    echo deleted: %%f
  )
)

pause
