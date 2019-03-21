# Linked Autoencoders Keras









---

This repository is a branch from a Private repository.

[Retaining History When Moving Files Across Repositories in Git](https://stosb.com/blog/retaining-history-when-moving-files-across-repositories-in-git/)

```bash
git clone https://github.com/chAwater/iDeepBio.git tmp
cd tmp
git remote rm origin
git filter-branch --subdirectory-filter Autoencoders_in_Keras -- --all
cd ../Linked_Autoencoders_Keras/
git remote add iDeepBio-branch ../tmp
git pull iDeepBio-branch master --allow-unrelated-histories
```
