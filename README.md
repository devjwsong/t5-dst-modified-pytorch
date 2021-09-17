# t5-dst-modified-pytorch
This is an unofficial implementation of the paper *Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking*[[1]](#1).

Additionally, this repository also contains the modified version of T5-dst, which takes multiple slot types as a single input by exploiting the pre-training object of the original T5[[2]](#2) model.

The first image describes the original T5-dst's procedure, using slot descriptions.

<img src="https://user-images.githubusercontent.com/16731987/133780757-582fbece-8754-486e-9328-b7729f8067f8.png" alt="The description of original T5-dst."/>

<br/>

And the second shows the modified version of T5-dst, inspired by T5's pre-training object.

<img src="https://user-images.githubusercontent.com/16731987/133783669-6628f87f-6e76-4c36-994c-2e6f87d2159c.png" alt="The description of T5-dst modified."/>

<br/>

---

### References

<a id="1">[1]</a> Lin, Z., Liu, B., Moon, S., Crook, P., Zhou, Z., Wang, Z., ... & Subba, R. (2021). Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking. *arXiv preprint arXiv:2105.04222*. ([https://arxiv.org/pdf/2105.04222.pdf](https://arxiv.org/pdf/2105.04222.pdf))

<a id="2">[2]</a> Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*. ([https://www.jmlr.org/papers/volume21/20-074/20-074.pdf](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf))
