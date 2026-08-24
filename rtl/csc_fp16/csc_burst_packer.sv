module csc_burst_packer (
  input logic clk, input logic rst_n,
  input logic record_valid, output logic record_ready, input logic [63:0] record_data,
  input logic flush,
  output logic burst_valid, input logic burst_ready,
  output logic [255:0] burst_data, output logic [2:0] burst_record_count
);
  logic [255:0] buffer; logic [2:0] count; logic pending;
  assign record_ready=!pending && count<4;
  assign burst_valid=pending;
  assign burst_data=buffer; assign burst_record_count=count;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin buffer<='0;count<=0;pending<=0; end
    else begin
      if(burst_valid&&burst_ready) begin buffer<='0;count<=0;pending<=0; end
      if(record_valid&&record_ready) begin
        buffer[count*64 +:64]<=record_data; count<=count+1'b1;
        if(count==3) pending<=1;
      end
      if(flush && count!=0 && !pending) pending<=1;
    end
  end
endmodule
