<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Q3_3.aspx.cs" Inherits="Q3_3" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
        選擇興趣：
    <asp:DropDownList ID="ddlHobby" runat="server">
        <asp:ListItem Text="--請選擇--" Value="" />
        <asp:ListItem Text="看書" Value="看書" />
        <asp:ListItem Text="打電腦" Value="打電腦" />
        <asp:ListItem Text="運動" Value="運動" />
    </asp:DropDownList><br />
    <asp:Button ID="btnShow" runat="server" Text="顯示所有資料" OnClick="btnShow_Click" /><br /><br />

    <asp:Label ID="lblAll" runat="server" Font-Size="Large"></asp:Label><br />

    <asp:RequiredFieldValidator ControlToValidate="ddlHobby" InitialValue=""
        ErrorMessage="*興趣必選" runat="server" />
    </form>
</body>
</html>
