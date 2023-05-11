# Probabilistic Abductive eXplanation


## (PAXp for decision trees)

The Python script `min_drset.py` implements the proposed methods in [iincms-corr22](https://arxiv.org/abs/2205.09569) for computing minimal $\delta$-relevant set (or MinPAXp) and locally-minimal $\delta$-relevant set (or ApproxPAXp).

### Usage illustration:

* Print the Usage:

<code>$> ./min_drset.py -h </code>

* Compute a MinPAXp:

<code>$> ./min_drset.py  -d 0.9 -i '3,9,1,3,9,11,5,1,1,5,3,11' -m ../tests/DTs/map/adult/IAI14/adult.map -t  ../tests/DTs/tree/adult/IAI14/adult.dt </code>

* Compute an ApproxPAXp:

<code>$> ./min_drset.py  -d 0.9 -w -i '3,9,1,3,9,11,5,1,1,5,3,11' -m ../tests/DTs/map/adult/IAI14/adult.map -t  ../tests/DTs/tree/adult/IAI14/adult.dt </code>

* Measure the explanation precision:

<code>$> ./min_drset.py  -x '2,3,8,9,10,11'  -i '3,9,1,3,9,11,5,1,1,5,3,11' -m ../tests/DTs/map/adult/IAI14/adult.map -t  ../tests/DTs/tree/adult/IAI14/adult.dt </code>

* Deciding if an ApproxPAXp is a PAXp:

<code>$> ./min_drset.py -d 0.9 -x '2,3,8,9,10,11'  -i '3,9,1,3,9,11,5,1,1,5,3,11' -m ../tests/DTs/map/adult/IAI14/adult.map -t  ../tests/DTs/tree/adult/IAI14/adult.dt </code>


## Citations

Please cite the following paper when you use this work:

```
@article{ixincms-ijar23,
  author       = {Yacine Izza and
                  Xuanxiang Huang and
                  Alexey Ignatiev and
                  Nina Narodytska and
                  Martin C. Cooper and
                  Jo{\~{a}}o Marques{-}Silva},
  title        = {On Computing Probabilistic Abductive Explanations},
  journal      = {International Journal of Approximate Reasoning},
  year         = {2023}
}

@article{iincms-corr22,
  author    = {Yacine Izza and
               Alexey Ignatiev and
               Nina Narodytska and
               Martin C. Cooper and
               Jo{\~{a}}o Marques{-}Silva},
  title     = {Provably Precise, Succinct and Efficient Explanations for Decision
               Trees},
  journal   = {CoRR},
  volume    = {abs/2205.09569},
  year      = {2022}
}

@article{ixincms-corr22,
  author       = {Yacine Izza and
                  Xuanxiang Huang and
                  Alexey Ignatiev and
                  Nina Narodytska and
                  Martin C. Cooper and
                  Jo{\~{a}}o Marques{-}Silva},
  title        = {On Computing Probabilistic Abductive Explanations},
  journal      = {CoRR},
  volume       = {abs/2212.05990},
  year         = {2022}
}

@article{iincms-corr21,
  author       = {Yacine Izza and
                  Alexey Ignatiev and
                  Nina Narodytska and
                  Martin C. Cooper and
                  Jo{\~{a}}o Marques{-}Silva},
  title        = {Efficient Explanations With Relevant Sets},
  journal      = {CoRR},
  volume       = {abs/2106.00546},
  year         = {2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.