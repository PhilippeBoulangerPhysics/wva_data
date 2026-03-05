

cd /home/philbou/scratch/isca_data/RT42_sst_m1/
for i in $(seq -f "run%04g" 1 200); do
  rm -r "$i"
done