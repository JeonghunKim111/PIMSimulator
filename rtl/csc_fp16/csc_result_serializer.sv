module csc_result_serializer (
  input logic in_valid, output logic in_ready,
  input logic [31:0] in_row, input logic [15:0] in_value,
  output logic out_valid, input logic out_ready, output logic [63:0] out_record
);
  assign in_ready=out_ready;
  assign out_valid=in_valid;
  assign out_record={16'b0,in_value,in_row};
endmodule
