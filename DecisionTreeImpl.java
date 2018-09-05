import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

/**
 * olllaa
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
  private DecTreeNode root;
  //ordered list of class labels
  private List<String> labels; 
  //ordered list of attributes
  private List<String> attributes; 
  //map to ordered discrete values taken by attributes
  private Map<String, List<String>> attributeValues; 
  
  /**
   * Answers static questions about decision trees.
   */
  DecisionTreeImpl() {

    // no code necessary this is void purposefully
  }

  /**
   * Build a decision tree given only a training set.
   * 
   * @param train: the training set
   */
  DecisionTreeImpl(DataSet train) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    this.root = ID3(train.instances,attributes, null, mostLabel(train.instances));
  }

  /**
   * Build a decision tree given a training set then prune it using a tuning set.
   * 
   * @param train: the training set
   * @param tune: the tuning set
   */
  DecisionTreeImpl(DataSet train, DataSet tune) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
  
    this.root = ID3(train.instances,attributes,null, mostLabel(train.instances));
    // prune the existing tree
    prune(root,tune);
    
    
 }
  
  private void prune(DecTreeNode root, DataSet tune){
	  
	  // 1. Compute T’s accuracy on TUNE; call it Acc(T) 
	   double accTune = getAccuracy(tune);
	   double hello = -999999;
	   boolean prune = true;  
	   // prune until no more improvementsgon
	   while(prune){
		   DecTreeNode bestNode = findBestPruningNode(root,tune);
		   
		   // test the best Node
		   bestNode.terminal = true;
	   	   hello = getAccuracy(tune);
	   	   bestNode.terminal=false; 
	   	   
	   	   if(hello<accTune){
	   		   // if bestNodeTo prune doesnt return the best accuracy stop pruning
	   		   prune = false;
	   	   }else{
	   		// if bestNodeToPrune returns a better accuracy when pruned, prune it from tree
	   		   accTune = hello;
	   		   // prune it from tree
	   		   bestNode.terminal=true;
	   	   }
	   }
  }
  
  private DecTreeNode findBestPruningNode(DecTreeNode root, DataSet tune)
	{
	  	// run BFS traversal. Try pruning each node and recording accuracy.
	  
	    double bestAcc=-999999;
		DecTreeNode nodeToPrune = root;
		Queue<DecTreeNode> queue = new LinkedList<DecTreeNode>();
		queue.add(root);
		while(!queue.isEmpty()){ 
			DecTreeNode node = queue.remove(); 
			if (!node.terminal){
				// set terminal to true and test accuracy using tuning set
				node.terminal = true;
				double currAcc = getAccuracy(tune);
				// check if accuracy is higher than previously recorded accuracy
				if (bestAcc <= currAcc){
					// if yes record the best accuracy
					// record node as best node to prune up till this point
					bestAcc = currAcc;
					nodeToPrune = node;
				}
				// re
				node.terminal = false;
			}
			//dont check for children if children is null
			if(node.children!=null){
				for(int i=0;i<node.children.size();i++){
					queue.add(node.children.get(i));
				}
			}
		
		}
		// return the node with the best accuracy
		return nodeToPrune;
	}


  @Override
  public String classify(Instance instance) {
	  DecTreeNode classify = root;
	  while(!classify.terminal){
		  int lol = getAttributeIndex(classify.attribute); 
		  int attrIndex = getAttributeValueIndex(classify.attribute,instance.attributes.get(lol));
		  classify = classify.children.get(attrIndex);
	
		  
	  }
	  
	  return classify.label;
  }

  @Override
  public void rootInfoGain(DataSet train ) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    // count labels
   
   
    //main entropy 
      // calc info gain for each attribute
    for(String attribute : attributes){
    	
    	 
    	 int [] labelCounter = new int[labels.size()];
    	 for(Instance example : train.instances){
    	    	labelCounter[getLabelIndex(example.label)]++;
    	    }
    	 double mainEntropy = calcEntropy(labelCounter);
    	// for each attribute, count the info gain using counter
    	double currGain = infoGain(mainEntropy,attribute,train.instances);
    	
		System.out.print(attribute);
		System.out.format(" %.5f\n", currGain);
    
    }
    
    
  }
  
  private String mostLabel(List<Instance> examples){
	  
	  int [] labelCounter = new int[labels.size()];
	  for(Instance exp :examples){
		  labelCounter[getLabelIndex(exp.label)]++;
	  }
	  
	  int bestIndex = 0;
	  int best = 0;
	  
	  for(int i = 0; i < labelCounter.length ; i++){
		  if(labelCounter[i] == labels.size()){
			  return labels.get(i);
		  }
		  
		  if(labelCounter[i]>best){																////////////////////////////////////perhaps semantics here??
			  best = labelCounter[i];
			  bestIndex = i;
		  }
		  
	  }
	  
	  return labels.get(bestIndex);
  }
  
  private DecTreeNode ID3(List<Instance>examples, List<String>attributes,String parentAttrVal, String defaultLabel){
	  int counter [] = new int[labels.size()];
	  //	  if empty(examples) then return default-label
	  if(examples.isEmpty()){
		  return new DecTreeNode (defaultLabel, null, parentAttrVal, true);
	  }
	 
	 
	  for(Instance example : examples){
		  counter[getLabelIndex(example.label)]++;
	  }
	  

	  
	  for(int i =0;i<counter.length;i++){
		  if(counter[i] == examples.size()){
			  return new DecTreeNode (labels.get(i), null, parentAttrVal, true);
		  }

	  }
	  
	  String majoritylabel = mostLabel(examples);
//	  if empty(attributes) then return majority-class of examples 
	  if(attributes.isEmpty()){
		  return new DecTreeNode(majoritylabel,null,parentAttrVal,true);
	  }
	  
//	  q = maxInfoGain(examples, attributes)
	  String bestAttr = maxInfoGainAttr(examples,attributes);
	  int indexOfBestAttr = getAttributeIndex(bestAttr);

	  List <String> newAttributes = new ArrayList<String>();
	  for(String attribute : attributes){
		  if(attribute.compareTo(bestAttr)!=0){
			  newAttributes.add(attribute);
		  }
	  }
	
//	  tree = create-node with attribute q  
	  DecTreeNode parent = new DecTreeNode(majoritylabel, bestAttr, parentAttrVal, false);
//	  foreach value v of attribute q do
	
		for(int i = 0; i<this.attributeValues.get(bestAttr).size();i++){
			
//		  v-ex = subset of examples with q == v
		  List <Instance> newExamples = new ArrayList<Instance>();
		  for(Instance example : examples){
			  if( (example.attributes.get(indexOfBestAttr)).compareTo(attributeValues.get(bestAttr).get(i))==0){
				  newExamples.add(example);
			  }
		  }
//		  subtree = buildtree(v-ex, attributes - {q}, majority-class(examples)) 
//		  add arc from tree to subtree
		  parent.addChild( ID3(newExamples, newAttributes, attributeValues.get(bestAttr).get(i), majoritylabel));
		 
	  }


	  return parent;
  }
  
  private String maxInfoGainAttr(List<Instance>examples,List<String>attr){

	  int [] labelCounter = new int[labels.size()];
	    for(Instance example : examples){
	    	labelCounter[getLabelIndex(example.label)]++;
	    }
	    //main entropy 
	    double mainEntropy = calcEntropy(labelCounter);
	    double infoGain = -1;
	    String best = null;
	    // calc info gain for each attribute
		    for(int i = 0;i<attr.size();i++){

	    	// for each attribute, count the info gain using counter
	    	double currGain = infoGain(mainEntropy,attr.get(i),examples);
	    	if(currGain>infoGain){
	    			infoGain = currGain;
	    			best =  attr.get(i);
	    	}
	    }
	  
	  return best;
  }
  private double infoGain(double mainEntropy, String attribute, List<Instance> instances){
	  // calculate specific entropy
	  double specificEntropy = 0;

	  List <String> attrValues = attributeValues.get(attribute);
	  int index = getAttributeIndex(attribute);
	// for each attribute value of the attribute
	  for(String val : attrValues){
		  double labelCount = 0;
		  int countByAttr[] = new int [labels.size()];
		  for(Instance example : instances){
			  if(example.attributes.get(index).compareTo(val)==0){
				  countByAttr[getLabelIndex(example.label)]++;
				  labelCount++;
			  }
		  }
		  
		  // H(X|Y)       += countY/ numExamples * H(X)    
		  specificEntropy+= (labelCount/instances.size())*calcEntropy(countByAttr);
	  }
	  	  return mainEntropy - specificEntropy;
  }
  
  private double calcEntropy(int [] counter){
	  
	  double sum = 0;
	  for(int count : counter){
			  sum += count;
	  }
	  
	  
	  double entropy = 0;
	  // for each label calc entropy 
	  for(int count : counter){
		 if(count >0){
		 entropy += -((double)count/sum)*(Math.log((double)count/sum)/Math.log(2));
		 }
	  }
		
	  return entropy;
	  
  }
  
  
  @Override
  /**
   * Print the decision tree in the specified format
   */
  public void print() {
    printTreeNode(root, null, 0);
  }

  /**
   * Prints the subtree of the node with each line prefixed by 4 * k spaces.
   */
  public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < k; i++) {
      sb.append("    ");
    }
    String value;
    if (parent == null) {
      value = "ROOT";
    } else {
      int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
      value = attributeValues.get(parent.attribute).get(attributeValueIndex);
    }
    sb.append(value);
    if (p.terminal) {
      sb.append(" (" + p.label + ")");
      System.out.println(sb.toString());
    } else {
      sb.append(" {" + p.attribute + "?}");
      System.out.println(sb.toString());
      for (DecTreeNode child : p.children) {
        printTreeNode(child, p, k + 1);
      }
    }
  }

  /**
   * Helper function to get the index of the label in labels list
   */
  private int getLabelIndex(String label) {
    for (int i = 0; i < this.labels.size(); i++) {
      if (label.equals(this.labels.get(i))) {
        return i;
      }
    }
    return -1;
  }
 
  /**
   * Helper function to get the index of the attribute in attributes list
   */
  private int getAttributeIndex(String attr) {
    for (int i = 0; i < this.attributes.size(); i++) {
      if (attr.equals(this.attributes.get(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
   */
  private int getAttributeValueIndex(String attr, String value) {
    for (int i = 0; i < attributeValues.get(attr).size(); i++) {
      if (value.equals(attributeValues.get(attr).get(i))) {
        return i;
      }
    }
    return -1;
  }
  
  
/**
   /* Returns the accuracy of the decision tree on a given DataSet.
   */
  @Override
  public double getAccuracy(DataSet ds){
	  // count for cprrectly classified
	  double correctlyClassified=0;
	  
	  // go through all instances and classify each of them
	 for(Instance example : ds.instances){
		 
		 // check if they were classified correctly using existing label
		 if(example.label.compareTo(classify(example))==0){
			 correctlyClassified++;
		 }
	 }
	 // return accuracy 
	  return correctlyClassified/ds.instances.size();
  }
}
