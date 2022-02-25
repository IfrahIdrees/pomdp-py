while getopts a:m:n:d:e:g:w:r:p: flag
do
    case "${flag}" in
        a) agent_type=${OPTARG};;
        m) maxsteps=${OPTARG};;
        n) num_sims=${OPTARG};;
        d) d=${OPTARG};;
        e) e=${OPTARG};;
        g) gr=${OPTARG};;
        w) wp=${OPTARG};;
        r) qr=${OPTARG};;
        p) qp=${OPTARG};; 
    esac
done

# Run experiment
python viz_rewardvssim.py --agent_type ${agent_type} \
    --maxsteps ${maxsteps} \
    --num_sims ${num_sims} \
    --d ${d} \
    --e ${e} \
    --wp ${wp} \
    --gr ${gr} \
    --qr ${qr} \
    --qp ${qp} \