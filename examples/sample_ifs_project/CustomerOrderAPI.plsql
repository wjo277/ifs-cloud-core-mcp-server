package CustomerOrderAPI is

   procedure CreateOrder (
      customer_no_    in varchar2,
      description_    in varchar2 default null,
      order_type_     in varchar2 default 'STANDARD',
      currency_       in varchar2 default 'USD'
   );
   
   procedure ConfirmOrder (
      order_no_       in number
   );
   
   procedure CancelOrder (
      order_no_       in number,
      reason_         in varchar2 default null
   );
   
   procedure UpdateOrderStatus (
      order_no_       in number,
      new_status_     in varchar2
   );
   
   function GetOrderTotal (
      order_no_       in number
   ) return number;
   
   function GetOrderDiscount (
      order_no_       in number,
      discount_type_  in varchar2 default 'STANDARD'
   ) return number;
   
   function IsOrderComplete (
      order_no_       in number
   ) return boolean;
   
   procedure ProcessDailyOrders;
   
   procedure GenerateOrderReport (
      from_date_      in date,
      to_date_        in date,
      customer_no_    in varchar2 default null
   );

end CustomerOrderAPI;

package body CustomerOrderAPI is

   procedure CreateOrder (
      customer_no_    in varchar2,
      description_    in varchar2 default null,
      order_type_     in varchar2 default 'STANDARD',
      currency_       in varchar2 default 'USD'
   ) is
      order_no_       number;
      customer_rec_   customer%rowtype;
   begin
      -- Validate customer exists
      select * into customer_rec_
      from customer
      where customer_no = customer_no_;
      
      -- Generate new order number
      select customer_order_seq.nextval into order_no_ from dual;
      
      -- Insert new order
      insert into customer_order (
         order_no,
         customer_no,
         order_date,
         description,
         status,
         order_type,
         currency,
         total_amount
      ) values (
         order_no_,
         customer_no_,
         sysdate,
         description_,
         'PENDING',
         order_type_,
         currency_,
         0
      );
      
      commit;
      
   exception
      when no_data_found then
         raise_application_error(-20001, 'Customer not found: ' || customer_no_);
      when others then
         rollback;
         raise;
   end CreateOrder;
   
   function GetOrderTotal (
      order_no_       in number
   ) return number is
      total_amount_   number := 0;
   begin
      select nvl(sum(ol.quantity * ol.unit_price), 0)
      into total_amount_
      from order_line ol
      where ol.order_no = order_no_;
      
      return total_amount_;
   exception
      when no_data_found then
         return 0;
   end GetOrderTotal;
   
   procedure ProcessDailyOrders is
      cursor pending_orders is
         select order_no, customer_no, total_amount
         from customer_order
         where status = 'PENDING'
         and order_date >= trunc(sysdate)
         order by order_date;
         
      order_count_    number := 0;
   begin
      for order_rec in pending_orders loop
         begin
            -- Process each order
            if order_rec.total_amount > 1000 then
               -- High value orders need approval
               UpdateOrderStatus(order_rec.order_no, 'PENDING_APPROVAL');
            else
               -- Auto confirm small orders
               ConfirmOrder(order_rec.order_no);
            end if;
            
            order_count_ := order_count_ + 1;
            
         exception
            when others then
               -- Log error but continue processing
               log_error('Error processing order: ' || order_rec.order_no || 
                        ', Error: ' || sqlerrm);
         end;
      end loop;
      
      -- Log summary
      log_info('Processed ' || order_count_ || ' daily orders');
      
   end ProcessDailyOrders;

end CustomerOrderAPI;