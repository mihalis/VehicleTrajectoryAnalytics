.. Vehicle trajectory Analytics documentation master file, created by
   sphinx-quickstart on Mon Nov 13 19:35:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vehicle trajectory Analytics's documentation!
========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   vta

**Vehicle Trajectory Analytics**: analyze and visualize vehicle trajectories obtained from instrumented vehicles with GPS and radar devices. Vehicle Trajectory Analytics is a Python package that lets you calculate a number of trajectory-based measures related to safety, acceleration limits and driver comfort, traffic flow, and lane changing behavior. 


Citation info
-------------

A Framework for Validating Traffic Simulation Models at the Vehicle Trajectory Level, February 2017, FHWA-JPO-16-405.


.. figure::  _static/front_page.png
   :align:   center

Features
--------

Vehicle Trajectory Analytics is built on top of pandas and matplotlib and works with any vehicle trajectory dataset to calculate:


  * gaps between vehicles 
  * the preceding and following vehicle  
  * lane position 
  * lane changes per driver 
  * distributions of speed, acceleration, jerk 
  * time to colision by driver or the distribution of TTC values 
  * lane change aggresiveness metrics on a case by case basis or the distribution of those metrics in the dataset 
  * identify outliers in terms of acceleration, gap, or combinations of speed and acceleration
  * driver comfort expressed by the acceleration root mean square metric 
  * calculate flow, speed, and density from trajectory values 
  * visualize trajectories in a space-time diagram 
  * produce speed heatmaps based on trajectories 
  * procuce lane change rate heatmap that shows lane changing activity along a corridor
  * save the calculated metrics to a csv or excel file 
  * save the visualizations into png or other image formats 

Examples and demonstrations of these features are in the GitHub repo (see below).
More feature development details are in the `change log`_.


Installation
------------

Install VehicleTrajectoryAnalytics with pip:

.. code-block:: shell

    pip install VeTrAn



Examples
--------

For examples and demos, see the `examples`_ GitHub repo.


Support
-------

The `issue tracker`_ is at the `VeTrAn GitHub repo`_.


License
-------

The project is licensed under the MIT license.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _change log: https://github.com/mihalis/VehicleTrajectoryAnalytics/CHANGELOG.md
.. _examples: https://github.com/mihalis/VehicleTrajectoryAnalytics/examples
.. _issue tracker: https://github.com/mihalis/VehicleTrajectoryAnalytics/issues
.. _VeTrAn GitHub repo: https://github.com/mihalis/VehicleTrajectoryAnalytics
