mrFaceDetection
===============

Distributed face detection using Hadoop and JavaCV

This is a simple library demonstrating the analysis of the CommonCrawl dataset
through implementing the canonical Hadoop Hello World program, a simple word
counter.

To build
--------

You'll need to have Apache Ant (http://ant.apache.org/manual/install.html)
installed, and once you do, just run a:

# ant dist

This step will compile the libraries and Hadoop code into an Elastic MapReduce-
friendly JAR at dist/lib/HelloWorld.jar, suitable for use as a custom JAR-based
Elastic MapReduce workflow.

To run locally
--------------

You'll need to be running Hadoop, and if you don't have it installed, Cloudera
provides a useful set of OS-specific Hadoop packages which will make it easy.
Check out their site:

https://ccp.cloudera.com/display/SUPPORT/Downloads

Once you've got Hadoop installed, you can use the 'hadoop jar' task to execute
the tutorial code.  Here's the pattern:

hadoop jar <checkout location>/dist/lib/HelloWorld.jar org.commoncrawl.tutorial.HelloWorld <Amazon AWS access key ID> <Amazon AWS secret access key> <CommonCrawl crawl files to use as input> <HDFS output location>
