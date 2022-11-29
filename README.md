# pyCFTrackers with an Onmidirectional Scale Estimator
Python re-implementation of some correlation filter based tracker

This project is adapted from the [pyTracker project released by fengyang95](https://github.com/fengyang95/pyCFTrackers).

The improvement of pyTracker is as follows:
1. An (OSE) Omnidirectional Scale Estimator is added to replace the original scale estimator in DCF-CSR tracker;

You can find the working mechanism of this OSE in the technique report in [GoogleDrive](https://drive.google.com/drive/folders/1i-294Y137ySk-4afjPpypxodpGenpJPZ?usp=share_link)

The Omnidirectional Scale Estimator source codes locate at [MRScale_estimator.py](https://github.com/ShawnZou717/pyCFTrackers/blob/master/cftracker/MRScale_estimator.py). Details are explained below.

## Omnidirectional Scale Estimator
