python main.py library initiate --dividingthreshold 0.1 --librarylocation /home/sfo/data/proteomics/fastas/search-db

#tests
#python main.py library enzyme -enzyme disenzyme --cut T=0,K=1,J=1 --noncut K=TP --librarylocation ~/data/proteomics/fastas/search-db
#python main.py library modification --modification acetylation --composition H2C2O1 --aminoacids KCSTYR --librarylocation /home/sfo/data/proteomics/fastas/search-db
#python main.py library modification --modification acetylation --mass 42.0367 --aminoacids KCSTYR --librarylocation/home/sfo/data/proteomics/fastas/search-db

python main.py library list --librarylocation /home/sfo/data/proteomics/fastas/search-db

python main.py library proteome --librarylocation /home/sfo/data/proteomics/fastas/search-db --proteomefile /home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien-NoTremb.fasta --missedcleavages 1 --variablemods 0 --enzyme 0 --maxvmods 3
