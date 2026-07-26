### 버킷 삭제 ###
```
export AWS_REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text)
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CLUSTER_NAME="vlm-distillation"
export BUCKET=vlm-data-${ACCOUNT_ID}-${AWS_REGION}

aws s3 rm s3://$BUCKET --recursive   # 안의 객체 전부 삭제
aws s3api delete-bucket --bucket $BUCKET --region $AWS_REGION  # 그다음 버킷 삭제
```

### EKS 클러스터 삭제하기 ###

```
# 1) LB/PVC 먼저 (안 지우면 나중에 VPC 삭제가 막힘)
kubectl delete ingress --all --all-namespaces   
kubectl delete svc --all --all-namespaces
kubectl delete pvc --all --all-namespaces

# 2) 카펜터 노드 스스로 정리시키기 (있다면)
kubectl delete nodepool --all
kubectl delete ec2nodeclass --all
kubectl patch ec2nodeclass gpu --type=merge -p '{"metadata":{"finalizers":[]}}'

# 3) 클러스터 삭제
eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION --wait


aws iam list-roles \
  --query "Roles[?contains(RoleName,'$CLUSTER_NAME')].RoleName" --output table

aws iam list-policies --scope Local \
  --query "Policies[?PolicyName=='mnist-s3-access'].Arn" --output text
# 나오면, 필요 없으면 삭제
aws iam delete-policy --policy-arn arn:aws:iam::$ACCOUNT_ID:policy/mnist-s3-access

aws cloudformation list-stacks --region $AWS_REGION \
  --query "StackSummaries[?contains(StackName,'eksctl-$CLUSTER_NAME') && StackStatus!='DELETE_COMPLETE'].StackName" \
  --output table
```





### vpc 삭제 ###
```
CF_STACK=$(cat CF_STACK | awk '{print $2}')
aws cloudformation delete-stack --stack-name ${CF_STACK} --region $AWS_REGION
```
