cd /lustre/home/acct-clsyzs/clsyzs/SunJianle/BNMR/app_blood_2205/prj/data
for i in {1..22}
do
    /lustre/home/acct-clsyzs/clsyzs/myr/imp_sample/plink2 --bgen /lustre/home/acct-clsyzs/clsyzs/myr/UKimp/ukb_imp_chr${i}_v3.bgen ref-first --sample /lustre/home/acct-clsyzs/clsyzs/myr/UKimp/ukb47192_imp_chr21_v3_s487296.sample --keep id_all.txt -geno 0.1 -mind 0.05 -maf 0.05 -hwe 0.0001 --make-bed --out chr${i}
    /lustre/home/acct-clsyzs/clsyzs/myr/imp_sample/plink2 --bfile ../../data_all/chr${i} --glm no-firth allow-no-covars --adjust --pfilter 1e-12 --pheno ../../data_all/pheno_data_all.txt --pheno-name label --1 --out SNP_${i}
    tail -n +2 SNP_${i}.label.glm.logistic | sed s/^\ *//g | expand -t 1 | tr -s ' ' ' ' | cut -f 3 -d ' ' > SNP_slt_${i}.txt
    if test $(cat SNP_slt_${i}.txt | wc -l) -gt 0
      then /lustre/home/acct-clsyzs/clsyzs/myr/imp_sample/plink2 --bfile ../../data_all/chr${i} --extract SNP_slt_${i}.txt --export A --out out_chr${i} 
    fi
done