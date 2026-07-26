### 버킷 삭제 ###
```
export ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export REGION=ap-northeast-2
export AZ=ap-northeast-2b
export BUCKET=vlm-data-${ACCOUNT_ID}-${REGION}
```
```
aws s3 rm s3://$BUCKET --recursive   # 안의 객체 전부 삭제
aws s3api delete-bucket --bucket $BUCKET --region $REGION  # 그다음 버킷 삭제
```

### EKS 클러스터 삭제하기 ###

```
# 1) LB/PVC 먼저 (안 지우면 나중에 VPC 삭제가 막힘)
kubectl delete svc --all --all-namespaces
kubectl delete pvc --all --all-namespaces

# 2) 카펜터 노드 스스로 정리시키기 (있다면)
kubectl delete nodepool --all
kubectl delete ec2nodeclass --all

# 3) 클러스터 삭제
eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION --wait
```









### vpc 삭제 ###
```
CF_STACK=$(cat CF_STACK | awk '{print $2}')
aws cloudformation delete-stack --stack-name ${CF_STACK} --region $AWS_REGION
```
