const Web3 = require("web3");
const contractABI = require("./FraudAuditABI.json");
const contractAddress = "0xYourContractAddress"; // Replace with actual address

const web3 = new Web3("https://polygon-mumbai.g.alchemy.com/v2/YOUR_API_KEY");

const contract = new web3.eth.Contract(contractABI, contractAddress);
const privateKey = "0xYourPrivateKey"; // Keep safe!
const senderAddress = "0xYourWalletAddress";

async function logFraudToBlockchain(transactionId, userId, reason) {
    const txData = contract.methods.logFraud(transactionId, userId, reason).encodeABI();

    const tx = {
        to: contractAddress,
        data: txData,
        gas: 3000000,
    };

    const signedTx = await web3.eth.accounts.signTransaction(tx, privateKey);
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);

    console.log("Fraud Logged: ", receipt.transactionHash);
}

module.exports = logFraudToBlockchain;
