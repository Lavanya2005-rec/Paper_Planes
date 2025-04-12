// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudAuditLog {

    struct FraudRecord {
        string transactionId;
        string userId;
        string reason;
        uint256 timestamp;
    }

    FraudRecord[] public fraudRecords;

    event FraudLogged(string transactionId, string userId, string reason, uint256 timestamp);

    function logFraud(string memory _transactionId, string memory _userId, string memory _reason) public {
        FraudRecord memory newRecord = FraudRecord(_transactionId, _userId, _reason, block.timestamp);
        fraudRecords.push(newRecord);
        emit FraudLogged(_transactionId, _userId, _reason, block.timestamp);
    }

    function getAllFrauds() public view returns (FraudRecord[] memory) {
        return fraudRecords;
    }

    function getFraudCount() public view returns (uint) {
        return fraudRecords.length;
    }
}
