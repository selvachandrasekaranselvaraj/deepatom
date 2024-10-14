if [ -f 'neb.xyz' ];then
   rm 'neb.xyz'
fi

for i in {0..36}
do
  if [ $i -lt 10 ]; then
    i="0$i"
  else
    i=$i
  fi
if [ -d "${i}" ]; then
   cd $i
   vasp2xyz vasprun.xml
   n1=$(head -n 1 vasprun.xyz)
   n=$((n1 + 2))
   tail -n $n vasprun.xyz >> ../neb.xyz
   cd ../
fi
done

